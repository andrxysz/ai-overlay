# Detector de coisas com IA (YOLOv8)

Essa aplicação detecta coisas na sua tela (overlay) utilizando IA (Yolo V8) É recomendado ter um PC decente, com uma placa de vídeo para o melhor funcionamento. É necessário Python 3.10.5.

## Requisitos

- Windows
- Python 3.10.5 (ou superior)

## Instalação dos reqs:

1. Execute o arquivo `install.bat` (duplo clique ou terminal).
2. O script instala automaticamente tudo do `requirements.txt`.

Opcional (manual):

```bash
python -m pip install -r requirements.txt
```

## Como executar

Depois de instalar os requisitos:

```bash
python main.py
```

## Modelo

O projeto usa o modelo `yolov8n.pt` e inclui presets de classes como `Pessoas`, `Animais` e `Veiculos`.
