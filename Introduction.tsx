
import { ArrowDownRight, ArrowUpRight, Orbit } from "lucide-react";

export default function Introduction() {
  return (
    <main className="mission-page" id="mission-top">
      <header className="mission-nav">
        <a className="mission-brand" href="#mission-top" aria-label="Neural Nexus home">
          <span className="mission-brand-mark">
            <Orbit size={18} strokeWidth={1.6} />
          </span>
          <span>
            NEURAL NEXUS
            <small>AI OBSERVATORY / 001</small>
          </span>
        </a>

        <nav aria-label="Main navigation">
          <a href="#mission-brief">MISSION</a>
          <a href="#overview">OBSERVATORY</a>
          <a href="#topology">ARCHITECTURE</a>
          <a href="#experiments">EXPERIMENTS</a>
        </nav>

        <a className="mission-enter" href="#overview">
          ENTER LAB <ArrowUpRight size={14} />
        </a>
      </header>

      <section className="mission-hero" id="mission-brief">
        <div className="mission-hero-copy">
          <div className="mission-kicker">
            <span className="mission-kicker-line" />
            RESEARCH ENVIRONMENT
            <span>•</span>
            INTERACTIVE MACHINE LEARNING
          </div>

          <h1>
            Look inside
            <br />
            the <em>learning</em>
            <br />
            process.
          </h1>

          <p className="mission-lede">
            Neural Nexus is an interactive observatory for a small neural
            network built from the ground up. Follow information as it moves
            through the model, change its training conditions, and inspect
            the evidence behind every prediction.
          </p>

          <div className="mission-hero-actions">
            <a className="mission-button" href="#overview">
              ENTER THE OBSERVATORY <ArrowUpRight size={16} />
            </a>
            <a className="mission-text-link" href="#mission-explainer">
              HOW IT WORKS <ArrowDownRight size={16} />
            </a>
          </div>

          <div className="mission-hero-foot">
            <span>01 / UNDERSTAND</span>
            <span>02 / EXPERIMENT</span>
            <span>03 / OBSERVE</span>
          </div>
        </div>

      </section>

      <section className="mission-explainer" id="mission-explainer">
        <div className="chapter-marker">
          <span>01</span>
          <b>THE MISSION</b>
        </div>

        <div className="mission-explainer-main">
          <h2>
            Machine learning is easier to understand when you can{" "}
            <em>see the mechanics.</em>
          </h2>

          <div className="mission-explainer-copy">
            <p>
              Many machine-learning tools show a final prediction without
              showing the process that produced it. Neural Nexus takes the
              opposite approach: it makes the model's structure, parameters,
              learning curve, and classification behaviour part of the
              experience.
            </p>

            <p>
              The project uses a NumPy multilayer perceptron with two input
              features, a ReLU hidden layer, and a sigmoid output. Training is
              handled by the existing Python backend; this interface
              visualizes the model state returned by that backend.
            </p>
          </div>
        </div>

         
      </section>

       
    </main>
  );
}
