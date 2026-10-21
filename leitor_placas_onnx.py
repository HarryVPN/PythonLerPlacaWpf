# import sys
# other_user_packages = "C:/Users/rafael.rodrigues.RO/AppData/Local/Python/pythoncore-3.14-64/Scripts"
# sys.path.append(other_user_packages)

import glob
import os
import re
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from observabilidade import configurar_observabilidade, medir_etapa, registrar_falha

MODELO_PATH = os.getenv(
    "MODELO_ONNX_PATH",
    r".\runs\detect\train-3\weights\best.onnx",
)
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CONFIANCA_MINIMA = float(os.getenv("CONFIANCA_MINIMA", "0.60"))
IOU_NMS = float(os.getenv("IOU_NMS", "0.45"))
OCR_TOKEN = "511bbb1efabebb18e7d019980a9b00c3ce85fac0"

OCR_URL = "https://api.platerecognizer.com/v1/plate-reader/"
OCR_DATA = {"regions": "br"}

app = Flask(__name__)
CORS(app)
configurar_observabilidade(app)

sessao_onnx = ort.InferenceSession(
    MODELO_PATH,
    providers=["CPUExecutionProvider"],
)
entrada_onnx = sessao_onnx.get_inputs()[0]
nome_entrada = entrada_onnx.name


def obter_dimensoes_entrada():
    forma = entrada_onnx.shape
    altura = forma[2] if len(forma) == 4 and isinstance(forma[2], int) else 640
    largura = forma[3] if len(forma) == 4 and isinstance(forma[3], int) else 640
    return largura, altura


LARGURA_MODELO, ALTURA_MODELO = obter_dimensoes_entrada()
cap = cv2.VideoCapture(CAMERA_INDEX)

ultima_caixa = None
ultima_box_json = None
ultima_confianca = 0.0
TEXTOBASE = "Procurando..."
texto_placa_atual = TEXTOBASE
bloqueio_ocr = False
tempoDeLeitura = 0
placaParaSalvar = ""
processamento_lock = threading.Lock()


def placa_valida(placa):
    placa = placa.upper().replace("-", "").replace(" ", "")
    return bool(
        re.fullmatch(r"[A-Z]{3}[0-9]{4}", placa)
        or re.fullmatch(r"[A-Z]{3}[0-9][A-Z][0-9]{2}", placa)
    )


def LetraViraNumero(letra):
    mapa = {"O": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
    return mapa.get(letra.upper(), letra)


def NumeroViraLetra(numero):
    mapa = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
    return mapa.get(numero, numero)


def corrigir_placa(texto):
    placa = re.sub(r"[^A-Z0-9]", "", texto.upper())
    if len(placa) != 7:
        return placa

    caracteres = list(placa)
    for indice in range(3):
        caracteres[indice] = NumeroViraLetra(caracteres[indice])

    caracteres[3] = LetraViraNumero(caracteres[3])

    if caracteres[4].isdigit():
        caracteres[4] = LetraViraNumero(caracteres[4])
    else:
        caracteres[4] = NumeroViraLetra(caracteres[4])

    caracteres[5] = LetraViraNumero(caracteres[5])
    caracteres[6] = LetraViraNumero(caracteres[6])
    return "".join(caracteres)


@medir_etapa("imagem.preparar")
def preparar_entrada_onnx(imagem):
    altura_original, largura_original = imagem.shape[:2]
    escala = min(
        LARGURA_MODELO / largura_original,
        ALTURA_MODELO / altura_original,
    )
    nova_largura = max(1, round(largura_original * escala))
    nova_altura = max(1, round(altura_original * escala))

    redimensionada = cv2.resize(imagem, (nova_largura, nova_altura))
    tela = np.full(
        (ALTURA_MODELO, LARGURA_MODELO, 3),
        114,
        dtype=np.uint8,
    )
    margem_x = (LARGURA_MODELO - nova_largura) // 2
    margem_y = (ALTURA_MODELO - nova_altura) // 2
    tela[
        margem_y : margem_y + nova_altura,
        margem_x : margem_x + nova_largura,
    ] = redimensionada

    tensor = cv2.cvtColor(tela, cv2.COLOR_BGR2RGB)
    tensor = tensor.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(tensor), escala, margem_x, margem_y


@medir_etapa("onnx.detectar")
def detectar_placas_onnx(imagem):
    tensor, escala, margem_x, margem_y = preparar_entrada_onnx(imagem)
    with medir_etapa("onnx.inferencia"):
        saida = sessao_onnx.run(None, {nome_entrada: tensor})[0]
    predicoes = np.squeeze(saida)

    if predicoes.ndim == 1:
        predicoes = predicoes[np.newaxis, ...]
    if predicoes.ndim != 2:
        return []

    # Saida comum do YOLOv8: (4 + classes, quantidade_de_deteccoes).
    if predicoes.shape[0] <= 85 and predicoes.shape[0] < predicoes.shape[1]:
        predicoes = predicoes.T
    if predicoes.shape[1] < 5:
        return []

    coordenadas = predicoes[:, :4]
    classes = predicoes[:, 4:]
    ids_classes = np.argmax(classes, axis=1)
    confiancas = classes[np.arange(len(classes)), ids_classes]

    # ==============================================================
    # DEBUG: VAMOS VER O QUE O MODELO ESTÁ "ENXERGANDO"
    # ==============================================================
    if len(confiancas) > 0:
        confianca_maxima = np.max(confiancas)
        print(f"\n--- DEBUG IA ---")
        print(f"Confiança MÁXIMA que o modelo achou na imagem: {confianca_maxima:.4f} ({confianca_maxima*100:.1f}%)")
        print(f"Sua CONFIANCA_MINIMA atual exige: {CONFIANCA_MINIMA} ({CONFIANCA_MINIMA*100}%)")
        print(f"----------------\n")
    # ==============================================================

    mascara = confiancas >= CONFIANCA_MINIMA

    coordenadas = coordenadas[mascara]
    confiancas = confiancas[mascara]
    ids_classes = ids_classes[mascara]
    altura_original, largura_original = imagem.shape[:2]

    caixas_nms = []
    deteccoes = []
    for (centro_x, centro_y, largura, altura), confianca, classe_id in zip(
        coordenadas,
        confiancas,
        ids_classes,
    ):
        x1 = int((centro_x - largura / 2 - margem_x) / escala)
        y1 = int((centro_y - altura / 2 - margem_y) / escala)
        x2 = int((centro_x + largura / 2 - margem_x) / escala)
        y2 = int((centro_y + altura / 2 - margem_y) / escala)

        x1 = max(0, min(x1, largura_original - 1))
        y1 = max(0, min(y1, altura_original - 1))
        x2 = max(0, min(x2, largura_original))
        y2 = max(0, min(y2, altura_original))
        if x2 <= x1 or y2 <= y1:
            continue

        caixas_nms.append([x1, y1, x2 - x1, y2 - y1])
        deteccoes.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "confianca": float(confianca),
                "classe": int(classe_id),
            }
        )

    if not caixas_nms:
        return []

    indices = cv2.dnn.NMSBoxes(
        caixas_nms,
        [item["confianca"] for item in deteccoes],
        CONFIANCA_MINIMA,
        IOU_NMS,
    )
    if len(indices) == 0:
        return []

    resultado = [deteccoes[int(i)] for i in np.array(indices).reshape(-1)]
    return sorted(resultado, key=lambda item: item["confianca"], reverse=True)


@medir_etapa("imagem.filtrar")
def aplicar_filtro_placa(imagem):
    imagem = cv2.resize(imagem, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    desfocada = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, binaria = cv2.threshold(
        desfocada,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binaria


@medir_etapa("ocr.extrair")
def extrair_texto_placa(caminho_imagem):
    if not OCR_TOKEN:
        print("Defina a variavel PLATE_RECOGNIZER_TOKEN para usar o OCR.")
        return None

    try:
        with open(caminho_imagem, "rb") as arquivo:
            resposta = requests.post(
                OCR_URL,
                files={"upload": arquivo},
                headers={"Authorization": f"Token {OCR_TOKEN}"},
                data=OCR_DATA,
                timeout=10,
            )
        resposta.raise_for_status()
        resultados = resposta.json().get("results", [])
        if not resultados:
            return None

        texto = corrigir_placa(resultados[0].get("plate", ""))
        return texto if placa_valida(texto) else None
    except (OSError, requests.RequestException, ValueError) as erro:
        registrar_falha("ocr.extrair", erro)
        print(f"Erro no OCR: {erro}")
        return None


def DesenharCaixa(imagem, caixa, confianca):
    if caixa is None:
        return
    x1, y1, x2, y2 = caixa["xyxy"]
    cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        imagem,
        f"Placa {confianca:.2f}",
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def DesenharHora(imagem):
    texto = f"{datetime.now().strftime('%H:%M:%S')} | {texto_placa_atual}"
    cv2.putText(
        imagem,
        texto,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )


@medir_etapa("ocr.assincrono")
def processar_ocr_assincrono(recorte):
    global bloqueio_ocr, placaParaSalvar, tempoDeLeitura, texto_placa_atual

    if not processamento_lock.acquire(blocking=False):
        return

    bloqueio_ocr = True
    caminho = f"temp_ocr_{time.time_ns()}.jpg"
    try:
        imagem_filtrada = aplicar_filtro_placa(recorte)
        if not cv2.imwrite(caminho, imagem_filtrada):
            return

        placa = extrair_texto_placa(caminho)
        if placa:
            texto_placa_atual = placa
            placaParaSalvar = placa
            tempoDeLeitura = time.time()
    finally:
        bloqueio_ocr = False
        processamento_lock.release()
        if os.path.exists(caminho):
            os.remove(caminho)


def ResetarProcura():
    global placaParaSalvar, texto_placa_atual, ultima_caixa, ultima_box_json
    global ultima_confianca

    placaParaSalvar = ""
    texto_placa_atual = TEXTOBASE
    ultima_caixa = None
    ultima_box_json = None
    ultima_confianca = 0.0


def tarefa_ia(imagem):
    global ultima_caixa, ultima_box_json, ultima_confianca

    deteccoes = detectar_placas_onnx(imagem)
    if not deteccoes:
        return

    melhor = deteccoes[0]
    ultima_caixa = melhor
    ultima_box_json = melhor["xyxy"]
    ultima_confianca = melhor["confianca"]

    if bloqueio_ocr:
        return

    x1, y1, x2, y2 = melhor["xyxy"]
    recorte = imagem[y1:y2, x1:x2]
    if recorte.size:
        threading.Thread(
            target=processar_ocr_assincrono,
            args=(recorte.copy(),),
            daemon=True,
        ).start()


def gerar_frames():
    while True:
        sucesso, imagem = cap.read()
        if not sucesso:
            time.sleep(0.1)
            continue

        tarefa_ia(imagem)
        DesenharCaixa(imagem, ultima_caixa, ultima_confianca)
        DesenharHora(imagem)

        sucesso, buffer = cv2.imencode(".jpg", imagem)
        if sucesso:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )


@app.route("/")
def index():
    return (
        '<img src="/video_feed" style="max-width:100%">'
        '<p><a href="/get_placa">Consultar placa atual</a></p>'
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        gerar_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/get_placa")
def get_placa():
    return jsonify(
        placa=texto_placa_atual,
        box=ultima_box_json,
        confianca=ultima_confianca,
    )


@app.route("/salvar", methods=["POST"])
def salvar():
    if not placaParaSalvar:
        return jsonify(erro="Nenhuma placa disponivel"), 400

    os.makedirs("FotosPlacas", exist_ok=True)
    nome = f"{placaParaSalvar}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    caminho = os.path.join("FotosPlacas", nome)

    sucesso, imagem = cap.read()
    if not sucesso or not cv2.imwrite(caminho, imagem):
        return jsonify(erro="Nao foi possivel salvar a imagem"), 500

    return jsonify(placa=placaParaSalvar, arquivo=caminho)


@app.route("/reset", methods=["POST"])
def reset():
    ResetarProcura()
    return jsonify(status="ok")


import os
import time
import cv2
import numpy as np
from flask import request, jsonify

@app.route("/processar_frame", methods=["POST"])
def processar_frame():
    if "frame" not in request.files:
        return jsonify(erro="Envie o arquivo no campo 'frame'"), 400

    dados = np.frombuffer(request.files["frame"].read(), dtype=np.uint8)
    imagem = cv2.imdecode(dados, cv2.IMREAD_COLOR)

    if imagem is None:
        return jsonify(erro="Imagem invalida"), 400

    # ==============================================================
    # DEBUG PASSO 1: Salva a imagem exata que chegou no Python
    # ==============================================================
    cv2.imwrite("debug_imagem_recebida.jpg", imagem)

    # preparar_entrada_onnx já converte BGR para RGB uma vez.
    deteccoes = detectar_placas_onnx(imagem)
    texto_placa = None

    # 2. Se encontrou alguma placa, faz o OCR
    if deteccoes:
        melhor = deteccoes[0]
        x1, y1, x2, y2 = melhor["xyxy"]

        # Garante que as coordenadas são inteiros
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # ATENÇÃO: O recorte continua usando a imagem original (OpenCV padrão)
        recorte = imagem[y1:y2, x1:x2]

        # ... (resto do seu código de OCR continua igual) ...

        if recorte.size:
            # ==============================================================
            # DEBUG PASSO 2: Salva o recorte da placa antes do OCR
            # Veja se o modelo recortou a placa certinho!
            # ==============================================================
            cv2.imwrite("debug_recorte_placa.jpg", recorte)

            caminho_temp = f"temp_ocr_req_{time.time_ns()}.jpg"
            try:
                # Aplica o filtro e salva temporariamente
                imagem_filtrada = aplicar_filtro_placa(recorte)
                cv2.imwrite(caminho_temp, imagem_filtrada)

                # Extrai o texto da placa
                texto_placa = extrair_texto_placa(caminho_temp)
            finally:
                # Limpa a imagem temporária
                if os.path.exists(caminho_temp):
                    os.remove(caminho_temp)

    # 3. Retorna o JSON contendo a propriedade "placa" que o JS espera
    return jsonify(
        deteccoes=deteccoes,
        box=(dict(zip(("x1", "y1", "x2", "y2"), deteccoes[0]["xyxy"])) if deteccoes else None),
        placa=texto_placa  # Se não achou, vai retornar null, o que é o comportamento correto para frames sem carro
    )

def limpar_temporarios():
    for caminho in glob.glob("temp_ocr_*.jpg"):
        try:
            os.remove(caminho)
        except OSError:
            pass


if __name__ == "__main__":
    limpar_temporarios()
    app.run(host="0.0.0.0", port=5000, threaded=True)
