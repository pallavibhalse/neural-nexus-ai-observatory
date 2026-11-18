
import { useMemo, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { Line, OrbitControls, Stars } from "@react-three/drei";
import { Vector3 } from "three";
import {
  Activity,
  Orbit,
  ChartNoAxesCombined,
  ChevronDown,
  CircleHelp,
  Database,
  FlaskConical,
  Gauge,
  GitCompareArrows,
  Home,
  Layers3,
  Play,
  RotateCcw,
  Settings2,
  Sparkles,
  Zap,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import Introduction from "./Introduction";

const API = "http://127.0.0.1:8000";

type HistoryPoint = {
  epoch: number;
  loss: number;
  accuracy: number;
  test_loss?: number;
  test_accuracy?: number;
};

type Point = {
  x: number;
  y: number;
  label: number;
  probability: number;
};

type Boundary = {
  x: number;
  y: number;
  p: number;
};

type State = {
  trained: boolean;
  dataset?: string;
  config?: {
    epochs: number;
    hidden_size: number;
    learning_rate: number;
    test_size?: number;
    seed?: number;
  };
  metrics?: {
    loss: number;
    accuracy: number;
    samples: number;
    epoch: number;
    train_loss?: number;
    train_accuracy?: number;
    test_loss?: number;
    test_accuracy?: number;
    test_samples?: number;
  };
  test_metrics?: {
    loss: number;
    accuracy: number;
    samples: number;
  };
  history?: HistoryPoint[];
  points?: Point[];
  boundary?: Boundary[];
  network?: {
    input_size: number;
    hidden_size: number;
    output_size: number;
    sample: {
      input: number[][];
      hidden_activations: number[];
      output_probability: number;
      prediction: number;
    };
    input_hidden_weights: number[][];
    hidden_output_weights: number[];
  };
};

function fmt(value: number | undefined, digits = 3) {
  return value === undefined ? "—" : value.toFixed(digits);
}

function NetworkView({ state }: { state: State | null }) {
  const hidden = state?.network?.hidden_size ?? 8;
  const [selected, setSelected] = useState("hidden-0");

  const nodes = useMemo(() => {
    const result: {
      id: string;
      position: [number, number, number];
      layer: "input" | "hidden" | "output";
      index: number;
    }[] = [];

    result.push({
      id: "input-0",
      position: [-3.1, 0.9, 0],
      layer: "input",
      index: 0,
    });

    result.push({
      id: "input-1",
      position: [-3.1, -0.9, 0],
      layer: "input",
      index: 1,
    });

    for (let i = 0; i < hidden; i++) {
      const y = hidden === 1 ? 0 : 1.95 - (i * 3.9) / (hidden - 1);
      const z = Math.sin(i * 1.35) * 0.42;

      result.push({
        id: `hidden-${i}`,
        position: [0, y, z],
        layer: "hidden",
        index: i,
      });
    }

    result.push({
      id: "output-0",
      position: [3.1, 0, 0],
      layer: "output",
      index: 0,
    });

    return result;
  }, [hidden]);

  const connections = useMemo(() => {
    const links: {
      id: string;
      from: [number, number, number];
      to: [number, number, number];
      weight: number;
    }[] = [];

    const inputs = nodes.filter((node) => node.layer === "input");
    const hiddenNodes = nodes.filter((node) => node.layer === "hidden");
    const output = nodes.find((node) => node.layer === "output");

    inputs.forEach((input) => {
      hiddenNodes.forEach((hiddenNode) => {
        const weight =
          state?.network?.input_hidden_weights?.[input.index]?.[
            hiddenNode.index
          ] ?? 0;

        links.push({
          id: `${input.id}-${hiddenNode.id}`,
          from: input.position,
          to: hiddenNode.position,
          weight,
        });
      });
    });

    if (output) {
      hiddenNodes.forEach((hiddenNode) => {
        const weight =
          state?.network?.hidden_output_weights?.[hiddenNode.index] ?? 0;

        links.push({
          id: `${hiddenNode.id}-${output.id}`,
          from: hiddenNode.position,
          to: output.position,
          weight,
        });
      });
    }

    return links;
  }, [nodes, state]);

  const selectedNode =
    nodes.find((node) => node.id === selected) ?? nodes[2];

  const selectedActivation =
    selectedNode?.layer === "hidden"
      ? state?.network?.sample?.hidden_activations?.[selectedNode.index]
      : selectedNode?.layer === "output"
        ? state?.network?.sample?.output_probability
        : state?.network?.sample?.input?.[0]?.[selectedNode.index];

  const nodeColor = (
    layer: "input" | "hidden" | "output",
    index: number
  ) => {
    if (layer === "input") return "#42d7ff";
    if (layer === "output") return "#ffb45c";

    const activation =
      state?.network?.sample?.hidden_activations?.[index] ?? 0;

    return activation > 0 ? "#ffb45c" : "#53678f";
  };

  return (
    <div className="network-canvas network-canvas-3d">
      <div className="network-label label-input">
        INPUT LAYER <span>02 features</span>
      </div>

      <div className="network-label label-hidden">
        HIDDEN LAYER{" "}
        <span>
          {String(hidden).padStart(2, "0")} neurons · ReLU
        </span>
      </div>

      <div className="network-label label-output">
        OUTPUT <span>01 neuron · Sigmoid</span>
      </div>

      <Canvas
        camera={{ position: [6.2, 4.1, 8.5], fov: 43 }}
        dpr={[1, 1.7]}
        gl={{ antialias: true, alpha: true }}
      >
        <color attach="background" args={["#050b1d"]} />
        <ambientLight intensity={0.75} />

        <pointLight
          position={[-4, 3, 5]}
          intensity={32}
          color="#42d7ff"
        />

        <pointLight
          position={[3, -2, 4]}
          intensity={25}
          color="#ffb45c"
        />

        <pointLight
          position={[0, 3, -4]}
          intensity={18}
          color="#ff8b6b"
        />

        <Stars
          radius={38}
          depth={24}
          count={900}
          factor={2.1}
          saturation={0}
          fade
          speed={0.35}
        />

        <group>
          {connections.map((link) => {
            const strength = Math.min(Math.abs(link.weight), 1);
            const positive = link.weight >= 0;

            return (
              <Line
                key={link.id}
                points={[
                  new Vector3(...link.from),
                  new Vector3(...link.to),
                ]}
                color={positive ? "#42d7ff" : "#ff9a76"}
                transparent
                opacity={state ? 0.16 + strength * 0.72 : 0.28}
                lineWidth={state ? 0.65 + strength * 1.5 : 0.9}
              />
            );
          })}

          {nodes.map((node) => {
            const color = nodeColor(node.layer, node.index);

            const active =
              node.layer !== "hidden" ||
              (state?.network?.sample?.hidden_activations?.[
                node.index
              ] ?? 0) > 0;

            const radius = node.layer === "hidden" ? 0.105 : 0.17;

            return (
              <group key={node.id} position={node.position}>
                <mesh
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelected(node.id);
                  }}
                >
                  <sphereGeometry args={[radius * 1.8, 24, 24]} />
                  <meshBasicMaterial
                    color={color}
                    transparent
                    opacity={0.12}
                  />
                </mesh>

                <mesh
                  onClick={(event) => {
                    event.stopPropagation();
                    setSelected(node.id);
                  }}
                >
                  <sphereGeometry args={[radius, 28, 28]} />

                  <meshStandardMaterial
                    color={active ? color : "#33415f"}
                    emissive={active ? color : "#1c2945"}
                    emissiveIntensity={
                      selected === node.id ? 2.7 : 1.25
                    }
                    roughness={0.24}
                    metalness={0.18}
                  />
                </mesh>

                {selected === node.id && (
                  <mesh>
                    <torusGeometry
                      args={[radius * 1.75, 0.012, 8, 40]}
                    />
                    <meshBasicMaterial
                      color="#f4f8ff"
                      transparent
                      opacity={0.9}
                    />
                  </mesh>
                )}
              </group>
            );
          })}
        </group>

        <OrbitControls
          enablePan={false}
          minDistance={5.5}
          maxDistance={15}
          minPolarAngle={0.55}
          maxPolarAngle={2.45}
          autoRotate
          autoRotateSpeed={0.28}
          makeDefault
        />
      </Canvas>

      <div className="network-3d-hint">
        DRAG TO ROTATE <span>·</span> SCROLL TO ZOOM{" "}
        <span>·</span> SELECT A NEURON
      </div>

      <div className="network-inspector">
        <span className="inspector-kicker">SELECTED NODE</span>

        <b>
          {selectedNode?.layer.toUpperCase()} ·{" "}
          {String(selectedNode?.index + 1).padStart(2, "0")}
        </b>

        <span className="inspector-value">
          {selectedActivation === undefined
            ? "Awaiting model data"
            : `Activation / value: ${selectedActivation.toFixed(4)}`}
        </span>
      </div>
    </div>
  );
}

function DecisionMap({ state }: { state: State | null }) {
  const points = state?.points ?? [];
  const boundary = state?.boundary ?? [];

  const size = 300;
  const min = -2.5;
  const max = 2.5;

  const pos = (v: number) => ((v - min) / (max - min)) * size;

  return (
    <div className="decision-wrap">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="decision-svg"
        role="img"
        aria-label="Model decision boundary"
      >
        <defs>
          <linearGradient id="surface">
            <stop offset="0%" stopColor="#0d70ff" stopOpacity=".08" />
            <stop offset="100%" stopColor="#ffb45c" stopOpacity=".13" />
          </linearGradient>

          <pattern
            id="grid"
            width="30"
            height="30"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 30 0 L 0 0 0 30"
              fill="none"
              stroke="#1c3159"
              strokeWidth=".7"
            />
          </pattern>
        </defs>

        <rect width={size} height={size} fill="#070d20" />

        {boundary.map((p, i) => (
          <rect
            key={i}
            x={pos(p.x) - size / 45 / 2}
            y={size - pos(p.y) - size / 45 / 2}
            width={size / 45 + 0.4}
            height={size / 45 + 0.4}
            fill={p.p >= 0.5 ? "#ffb45c" : "#168bff"}
            opacity={0.05 + Math.abs(p.p - 0.5) * 0.22}
          />
        ))}

        <rect width={size} height={size} fill="url(#grid)" />

        {points.map((p, i) => (
          <circle
            key={i}
            cx={pos(p.x)}
            cy={size - pos(p.y)}
            r="2.4"
            fill={p.label === 1 ? "#ff9a76" : "#29dfff"}
            fillOpacity=".9"
            stroke="#dbeafe"
            strokeOpacity=".16"
            strokeWidth=".5"
          />
        ))}
      </svg>

      <div className="map-legend">
        <span>
          <i className="dot cyan" />
          Class 0
        </span>

        <span>
          <i className="dot pink" />
          Class 1
        </span>

        <span className="muted">2D feature space</span>
      </div>
    </div>
  );
}

export default function App() {
  const [dataset, setDataset] = useState("circles");
  const [epochs, setEpochs] = useState(500);
  const [hidden, setHidden] = useState(8);
  const [lr, setLr] = useState(0.03);

  const [state, setState] = useState<State | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [notice, setNotice] = useState(
    "Ready to initialize your network"
  );

  async function train() {
    setBusy(true);
    setError("");
    setNotice("Training network · calculating gradients");

    try {
      const response = await fetch(`${API}/api/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          dataset,
          epochs,
          hidden_size: hidden,
          learning_rate: lr,
          seed: 7,
        }),
      });

      if (!response.ok) {
        throw new Error(
          `API returned ${response.status}. Is the backend running?`
        );
      }

      const data = (await response.json()) as State;

      setState(data);
      setNotice("Training complete · real model data received");
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not connect to the backend"
      );

      setNotice("Backend connection needed");
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    setState(null);
    setError("");
    setNotice("Network reset · ready for a new experiment");
  }

  const history = state?.history ?? [];
  const latest = history[history.length - 1];

  const trainingAccuracy =
    state?.metrics?.train_accuracy ?? state?.metrics?.accuracy;

  const testAccuracy =
    state?.metrics?.test_accuracy ?? state?.test_metrics?.accuracy;

  const trainingLoss =
    state?.metrics?.train_loss ?? state?.metrics?.loss;

  const testLoss =
    state?.metrics?.test_loss ?? state?.test_metrics?.loss;

  const testSamples =
    state?.metrics?.test_samples ?? state?.test_metrics?.samples;

  return (
    <>
      <Introduction />

      <div className="app-shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">
              <Orbit size={25} />
            </div>

            <div>
              <b>
                NEURAL<span>NEXUS</span>
              </b>

              <small>AI OBSERVATORY / 001</small>
            </div>
          </div>

          <div className="side-caption">WORKSPACE</div>

          <nav>
            <a href="#overview" className="nav-item active">
              <Home size={17} />
              Overview
              <span className="nav-light" />
            </a>

            <a href="#topology" className="nav-item">
              <Orbit size={17} />
              Neural topology
            </a>

            <a href="#experiments" className="nav-item">
              <FlaskConical size={17} />
              Experiments
              <span className="nav-count">01</span>
            </a>

            <a href="#playground" className="nav-item">
              <Database size={17} />
              Data playground
            </a>

            <a href="#visualizations" className="nav-item">
              <ChartNoAxesCombined size={17} />
              Visualizations
            </a>
          </nav>

          <div className="sidebar-bottom">
            <p>
              Not just a model.
              <br />
              <strong>A window into how AI learns.</strong>
            </p>

            <div className="sidebar-version">
              NEURAL NEXUS <span>v0.1.0</span>
            </div>
          </div>
        </aside>

        <main className="main">
          <header className="topbar">
            <div className="breadcrumbs">
              WORKSPACE <span>/</span> <b>OVERVIEW</b>
            </div>

            <div className="top-status">
              <span
                className={`status-dot ${error ? "offline" : ""}`}
              />

              {error ? "API OFFLINE" : "LOCAL LAB"}

              <span className="top-divider" />
              <span className="avatar">NX</span>
            </div>
          </header>

          <section id="overview" className="page-heading">
            <div>
              <div className="eyebrow">
                <Sparkles size={13} />
                INTERACTIVE MACHINE LEARNING ENVIRONMENT
              </div>

              <h1>
                Neural <span>Nexus</span>
              </h1>

              <p>
                Explore the mechanics of learning.{" "}
                <b>See what the model sees.</b>
              </p>
            </div>

            <div className="session-pill">
              <span className="status-dot" />
              SESSION / LOCAL
              <small>● ENCRYPTED LOOPBACK</small>
            </div>
          </section>

          <section id="experiments" className="control-panel panel">
            <div className="control-title">
              <div className="section-icon">
                <Settings2 size={17} />
              </div>

              <div>
                <b>Experiment controls</b>
                <small>Configure the learning engine</small>
              </div>
            </div>

            <label className="control-field">
              <span>DATASET</span>

              <div className="select-wrap">
                <select
                  value={dataset}
                  onChange={(e) => setDataset(e.target.value)}
                >
                  <option value="circles">Concentric circles</option>
                  <option value="moons">Two moons</option>
                  <option value="xor">XOR pattern</option>
                </select>

                <ChevronDown size={14} />
              </div>
            </label>

            <label className="control-field">
              <span>
                HIDDEN NEURONS <b>{hidden}</b>
              </span>

              <input
                type="range"
                min="2"
                max="24"
                value={hidden}
                onChange={(e) => setHidden(Number(e.target.value))}
              />
            </label>

            <label className="control-field">
              <span>
                LEARNING RATE <b>{lr.toFixed(3)}</b>
              </span>

              <input
                type="range"
                min=".005"
                max=".2"
                step=".005"
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
              />
            </label>

            <label className="control-field">
              <span>
                TRAINING EPOCHS <b>{epochs}</b>
              </span>

              <input
                type="range"
                min="100"
                max="1500"
                step="100"
                value={epochs}
                onChange={(e) => setEpochs(Number(e.target.value))}
              />
            </label>

            <div className="control-actions">
              <button
                className="btn-primary"
                onClick={train}
                disabled={busy}
              >
                <Play size={15} fill="currentColor" />
                {busy ? "Training…" : "Train model"}
              </button>

              <button
                className="btn-ghost"
                onClick={reset}
                disabled={busy}
              >
                <RotateCcw size={14} />
                Reset
              </button>
            </div>
          </section>

          {error && (
            <div className="error-banner">
              <CircleHelp size={16} />

              <div>
                <b>Backend not connected</b>

                <span>
                  {error} Start FastAPI using{" "}
                  <code>
                    python -m uvicorn backend.main:app --reload
                  </code>{" "}
                  from the project root.
                </span>
              </div>
            </div>
          )}

          <section className="metrics-grid">
            <div className="metric-card panel">
              <div className="metric-top">
                <span>TRAINING ACCURACY</span>
                <Gauge size={16} />
              </div>

              <div className="metric-value">
                {trainingAccuracy === undefined
                  ? "—"
                  : `${(trainingAccuracy * 100).toFixed(1)}%`}
              </div>

              <div className="metric-foot">
                <span className="metric-bar">
                  <i
                    style={{
                      width: `${(trainingAccuracy ?? 0) * 100}%`,
                    }}
                  />
                </span>

                <span>Seen data</span>
              </div>
            </div>

            <div className="metric-card panel">
              <div className="metric-top">
                <span>TEST ACCURACY</span>
                <Activity size={16} />
              </div>

              <div className="metric-value">
                {testAccuracy === undefined
                  ? "—"
                  : `${(testAccuracy * 100).toFixed(1)}%`}
              </div>

              <div className="metric-foot">
                <span className="metric-bar">
                  <i
                    style={{
                      width: `${(testAccuracy ?? 0) * 100}%`,
                    }}
                  />
                </span>

                <span>Unseen data</span>
              </div>
            </div>

            <div className="metric-card panel">
              <div className="metric-top">
                <span>TRAINING LOSS</span>
                <Zap size={16} />
              </div>

              <div className="metric-value">
                {trainingLoss === undefined
                  ? "—"
                  : fmt(trainingLoss, 4)}
              </div>

              <div className="metric-foot">
                <span className="cyan-text">
                  Binary cross-entropy
                </span>

                <span>↓ better</span>
              </div>
            </div>

            <div className="metric-card panel">
              <div className="metric-top">
                <span>TEST LOSS</span>
                <Database size={16} />
              </div>

              <div className="metric-value">
                {testLoss === undefined ? "—" : fmt(testLoss, 4)}
              </div>

              <div className="metric-foot">
                <span className="violet-text">
                  {testSamples ?? "—"} test samples
                </span>

                <span>↓ better</span>
              </div>
            </div>
          </section>

          <section id="visualizations" className="primary-grid">
            <div id="topology" className="panel network-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    LIVE MODEL INSPECTION
                  </span>

                  <h2>Neural topology</h2>
                </div>

                <span className="live-tag">
                  <i />
                  {state ? "MODEL TRAINED" : "AWAITING MODEL"}
                </span>
              </div>

              <NetworkView state={state} />

              <div className="network-footer">
                <span>
                  <i className="dot cyan" />
                  Input features
                </span>

                <span>
                  <i className="dot violet" />
                  ReLU hidden layer
                </span>

                <span>
                  <i className="dot pink" />
                  Sigmoid output
                </span>
              </div>
            </div>

            <div className="panel loss-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    OPTIMIZATION TRACE
                  </span>

                  <h2>Learning curve</h2>
                </div>

                <span className="chart-legend">
                  <i />
                  TRAIN / TEST LOSS
                </span>
              </div>

              <div className="chart-stat">
                <span>
                  Training:{" "}
                  <b>{fmt(latest?.loss ?? trainingLoss, 4)}</b>
                  {"  "}·{"  "}
                  Test:{" "}
                  <b>{fmt(latest?.test_loss ?? testLoss, 4)}</b>
                </span>
              </div>

              <div className="chart-area">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={history}>
                    <defs>
                      <linearGradient
                        id="lossFill"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor="#24d8ff"
                          stopOpacity=".26"
                        />

                        <stop
                          offset="100%"
                          stopColor="#24d8ff"
                          stopOpacity="0"
                        />
                      </linearGradient>

                      <linearGradient
                        id="testLossFill"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor="#ffb45c"
                          stopOpacity=".20"
                        />

                        <stop
                          offset="100%"
                          stopColor="#ffb45c"
                          stopOpacity="0"
                        />
                      </linearGradient>
                    </defs>

                    <CartesianGrid
                      stroke="#182746"
                      strokeDasharray="3 5"
                    />

                    <XAxis
                      dataKey="epoch"
                      tick={{ fill: "#7284a9", fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                    />

                    <YAxis
                      tick={{ fill: "#7284a9", fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                      width={36}
                    />

                    <Tooltip
                      contentStyle={{
                        background: "#0b1228",
                        border: "1px solid #1a3c62",
                        borderRadius: 8,
                        color: "#eaf5ff",
                      }}
                    />

                    <Area
                      type="monotone"
                      dataKey="loss"
                      name="Training loss"
                      stroke="#28dfff"
                      strokeWidth={2}
                      fill="url(#lossFill)"
                      isAnimationActive={true}
                    />

                    <Area
                      type="monotone"
                      dataKey="test_loss"
                      name="Test loss"
                      stroke="#ffb45c"
                      strokeWidth={2}
                      fill="url(#testLossFill)"
                      isAnimationActive={true}
                      connectNulls={false}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="chart-foot">
                <span>
                  <i className="dot cyan" />
                  Training loss
                </span>

                <span>
                  <i className="dot violet" />
                  Test loss
                </span>

                <span>Recorded {history.length} points</span>
              </div>
            </div>
          </section>

          <section className="secondary-grid">
            <div
              id="playground"
              className="panel decision-panel"
            >
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    CLASSIFICATION SPACE
                  </span>

                  <h2>Decision boundary</h2>
                </div>

                <span className="small-tag">2D / LIVE</span>
              </div>

              <DecisionMap state={state} />

              <p className="panel-note">
                The background represents model confidence; points
                show the generated dataset.
              </p>
            </div>

            <div className="panel experiment-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">RUN SUMMARY</span>
                  <h2>Experiment snapshot</h2>
                </div>

                <GitCompareArrows
                  size={18}
                  className="subtle-icon"
                />
              </div>

              <div className="snapshot-row">
                <span>Dataset</span>
                <b>{state?.dataset ?? dataset}</b>
              </div>

              <div className="snapshot-row">
                <span>Hidden layer</span>
                <b>
                  {state?.network?.hidden_size ?? hidden} neurons ·
                  ReLU
                </b>
              </div>

              <div className="snapshot-row">
                <span>Optimizer</span>
                <b>Batch gradient descent</b>
              </div>

              <div className="snapshot-row">
                <span>Learning rate</span>
                <b>
                  {(state?.config?.learning_rate ?? lr).toFixed(3)}
                </b>
              </div>

              <div className="snapshot-row">
                <span>Train / test split</span>
                <b>
                  {state?.metrics?.samples ?? "—"} /{" "}
                  {testSamples ?? "—"}
                </b>
              </div>

              <div className="insight-box">
                <div className="insight-icon">
                  <Sparkles size={15} />
                </div>

                <div>
                  <b>What is happening?</b>

                  <p>
                    {state
                      ? `The model achieved ${
                          trainingAccuracy === undefined
                            ? "—"
                            : `${(trainingAccuracy * 100).toFixed(1)}%`
                        } training accuracy and ${
                          testAccuracy === undefined
                            ? "—"
                            : `${(testAccuracy * 100).toFixed(1)}%`
                        } test accuracy. Compare both values to understand how well it generalizes to unseen data.`
                      : "Initialize a run to inspect real activations, weights, loss, and predictions from the NumPy model."}
                  </p>
                </div>
              </div>

              <button
                className="btn-secondary"
                onClick={train}
                disabled={busy}
              >
                <Play size={13} />
                Run experiment
                <span>↗</span>
              </button>
            </div>

            <div className="panel progress-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">
                    LEARNING PIPELINE
                  </span>

                  <h2>Inside the loop</h2>
                </div>

                <Layers3
                  size={18}
                  className="subtle-icon"
                />
              </div>

              {[
                "Initialize weights",
                "Forward propagation",
                "Calculate loss",
                "Backpropagation",
                "Update parameters",
              ].map((step, i) => (
                <div className="pipeline-step" key={step}>
                  <span
                    className={`step-index ${state ? "done" : ""}`}
                  >
                    {state
                      ? "✓"
                      : String(i + 1).padStart(2, "0")}
                  </span>

                  <div>
                    <b>{step}</b>

                    <small>
                      {
                        [
                          "He initialization",
                          "Inputs → activations",
                          "Binary cross-entropy",
                          "Compute gradients",
                          "Gradient descent",
                        ][i]
                      }
                    </small>
                  </div>

                  <span className="step-state">
                    {state ? "DONE" : "READY"}
                  </span>
                </div>
              ))}
            </div>
          </section>

          <footer className="footer">
            <span>
              <i
                className={`status-dot ${error ? "offline" : ""}`}
              />
              {notice}
            </span>

            <span>
              PYTHON + NUMPY <i /> FASTAPI <i /> REACT <i /> TYPESCRIPT
            </span>

            <span>BUILD · EXPERIMENT · UNDERSTAND</span>
          </footer>
        </main>
      </div>
    </>
  );
}
