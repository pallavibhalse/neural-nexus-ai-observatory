# Neural Nexus — From Scratch

A futuristic, interactive neural-network laboratory. The model is implemented with NumPy (no ML framework for the learning engine), served through FastAPI, and visualized in a React + TypeScript frontend.

## Features
- NumPy multilayer perceptron with manual forward propagation, backpropagation, and gradient descent
- Synthetic `XOR`, `Circles`, and `Moons` datasets
- Dark neon dashboard with network visualization, training metrics, loss chart, and decision boundary
- Adjustable learning rate, hidden-layer width, and training epochs
- API health endpoint and unit tests

## Requirements
- Python 3.10+
- Node.js 18+

## Run on Windows PowerShell

### 1. Backend
From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Backend API: http://127.0.0.1:8000  
API docs: http://127.0.0.1:8000/docs

### 2. Frontend
Open a second terminal in the project root:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL Vite prints, usually http://localhost:5173.

## Project structure

```text
neural_nexus/
├── backend/
│   └── main.py
├── src/
│   ├── __init__.py
│   ├── neural_network.py
│   └── datasets.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       └── styles.css
├── tests/
│   └── test_neural_network.py
├── requirements.txt
└── README.md
```

## Notes
- The first version trains synchronously when you press **Train model**. The UI displays the resulting training history after the API returns.
- This is an educational binary-classification lab, not a production model-serving system.
