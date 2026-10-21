# Observabilidade do leitor ONNX

O `leitor_placas_onnx.py` usa `observabilidade.py` para enviar métricas, logs e traces para a stack local do BOL. O serviço aparece como **BOL-Python**.

## Executar

No ambiente Python que você já usa para executar o leitor (Python 3.10+), instale:

```powershell
python -m pip install -r requirements-observability.txt
```

Na pasta `BOL-prod`, suba a stack:

```powershell
.\scripts\start-local.ps1 -StackOnly
```

Depois, na pasta `Python couser`, execute normalmente:

```powershell
python leitor_placas_onnx.py
```

O modelo ONNX e as dependências já utilizadas pelo leitor continuam sendo necessários. A configuração assume desenvolvimento local. Se o ambiente `venv` apontar para um Python que não existe mais, selecione um interpretador válido antes de instalar as dependências; não é necessário alterar o ambiente do BOL.

- Dashboard Python: http://localhost:3000/d/bol-python
- Jaeger: http://localhost:16686 — selecione `BOL-Python`.
- Prometheus: http://localhost:9090 — filtre `service_name="BOL-Python"`.

Reinicie o Grafana após adicionar os arquivos de provisionamento se a stack já estava aberta. Gere algumas requisições e aguarde cerca de 15 segundos. O Collector recebe em `http://localhost:4318`; não é necessário expor `/metrics` no Flask.

## Medidas

O painel mostra chamadas, duração média, p95 e respostas 4xx/5xx para todas as rotas HTTP do Flask: `/`, `/video_feed`, `/get_placa`, `/salvar`, `/reset` e `/processar_frame`.

As etapas têm histogramas de duração e contadores de falhas: `imagem.preparar`, `onnx.detectar`, `onnx.inferencia`, `imagem.filtrar`, `ocr.extrair` e `ocr.assincrono`. `onnx.detectar` inclui preparação, inferência e pós-processamento: não some os tempos de spans aninhados como se fossem independentes.

Falhas de OCR que o código já captura e converte em `None` também são registradas, sem mudar o retorno. Imagem sem placa não é considerada falha. O tempo de `/video_feed` não representa o tempo de um frame: é uma resposta contínua; use as etapas de detecção/OCR para esse fluxo.

O `traceparent` recebido do BOL é propagado para o Python, para `requests` e para threads de trabalho. Para um trace completo, o BOL precisa chamar esta instância instrumentada do Python, e ambos devem enviar à mesma stack. Endereços de API existentes no BOL não foram alterados.

Os novos logs registram somente rota, status, duração, etapa e tipo da falha. Não exportam `print`, imagens, placas reconhecidas ou cabeçalhos de autorização. Spans automáticos das bibliotecas podem registrar URLs e detalhes de exceções; use dados de teste e não habilite captura de headers sensíveis.

## Desligar

```powershell
$env:BOL_OBSERVABILITY_ENABLED = 'false'
python leitor_placas_onnx.py
```

Para reativar, defina `BOL_OBSERVABILITY_ENABLED=true`. Fora do desenvolvimento, defina `BOL_ENVIRONMENT=Production`; a instrumentação não será inicializada. O endpoint opcional `BOL_OTLP_ENDPOINT` aceita apenas HTTP em localhost/loopback. Reinicie o processo após mudar essas variáveis.

A coleta acontece em segundo plano. Com a stack desligada, o Python continua funcionando, mas a telemetria pode ser perdida e os exportadores podem escrever erros no console. Todos os traces locais são coletados; meça o overhead antes de comparar desempenho, sem supor ganho pela instrumentação.

## Teste sem câmera

```powershell
python test_observabilidade.py
```

Usa Flask e um receptor OTLP locais. Verifica os três sinais, métricas, erros, correlação entre serviços/threads e ausência do Collector. O traceback de `synthetic failure`/`private-value` no console é intencional no cenário de erro. O teste não carrega ONNX nem faz chamadas à API OCR.