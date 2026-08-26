"use client";

import { useState } from "react";
import { Bot, Footprints, Route } from "lucide-react";
import type { LearningExperience } from "../../../lib/learning";
import {
  discountedReturn,
  epsilonGreedyDistribution,
  expectedFlipsForConsecutiveHeads,
  trajectoryProbability,
} from "../../../lib/probabilityLearning";
import {
  ProbabilityCheck,
  ProbabilityCourse,
  ProbabilityFormula,
  ProbabilityInsight,
  ProbabilityMetric,
  ProbabilityQuizLaunch,
  ProbabilitySection,
  probabilityCourseStyles as styles,
} from "../probability/ProbabilityCourse";

type Props = { experience: LearningExperience };

const ACTIONS = [
  { name: "up", symbol: "↑", delta: [-1, 0] },
  { name: "right", symbol: "→", delta: [0, 1] },
  { name: "down", symbol: "↓", delta: [1, 0] },
  { name: "left", symbol: "←", delta: [0, -1] },
] as const;

const VALUES = [
  [2.7, 3.2, 4.2, 0],
  [2.2, 2.4, 3.1, 4.2],
  [1.7, 1.5, -2, 2.9],
  [1.2, 1.0, 1.3, 2.0],
] as const;

function HeroDecisionWorld() {
  return (
    <div
      className={styles.lab}
      aria-label="A state action reward next-state loop"
    >
      <div className={styles.labHeader}>
        <div>
          <p>Decision world</p>
          <h3>Prediction changes what happens next</h3>
        </div>
        <Bot aria-hidden="true" size={35} />
      </div>
      <div className={styles.formulaTrail}>
        <div className={styles.formulaStep}>
          <strong>state</strong>
          <span>where am I?</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>action</strong>
          <span>what do I try?</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>transition</strong>
          <span>where do I land?</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>reward</strong>
          <span>what feedback arrives?</span>
        </div>
        <div className={styles.formulaStep}>
          <strong>repeat</strong>
          <span>now the future changed</span>
        </div>
      </div>
    </div>
  );
}

type Position = readonly [number, number];

function move([row, column]: Position, actionIndex: number): Position {
  const delta = ACTIONS[actionIndex]?.delta ?? [0, 0];
  return [
    Math.max(0, Math.min(3, row + delta[0])),
    Math.max(0, Math.min(3, column + delta[1])),
  ];
}

function cellReward([row, column]: Position) {
  if (row === 0 && column === 3) return 5;
  if (row === 2 && column === 2) return -4;
  return -0.1;
}

function GridworldLab() {
  const [actionIndex, setActionIndex] = useState(0);
  const [gamma, setGamma] = useState(0.9);
  const current: Position = [2, 1];
  const outcomeActions = [
    actionIndex,
    (actionIndex + 3) % 4,
    (actionIndex + 1) % 4,
  ];
  const probabilities = [0.8, 0.1, 0.1];
  const outcomes = outcomeActions.map((actualAction, index) => {
    const next = move(current, actualAction);
    const reward = cellReward(next);
    const value = VALUES[next[0]]?.[next[1]] ?? 0;
    return {
      actualAction,
      next,
      probability: probabilities[index] ?? 0,
      reward,
      value,
    };
  });
  const expectedReward = outcomes.reduce(
    (sum, outcome) => sum + outcome.probability * outcome.reward,
    0,
  );
  const qEstimate = outcomes.reduce(
    (sum, outcome) =>
      sum + outcome.probability * (outcome.reward + gamma * outcome.value),
    0,
  );
  const highlighted = new Set(outcomes.map(({ next }) => next.join("-")));

  return (
    <div className={styles.lab} data-testid="gridworld-lab">
      <div className={styles.labHeader}>
        <div>
          <p>Stochastic 4 × 4 grid</p>
          <h3>You choose an action; the environment chooses the outcome</h3>
        </div>
        <Route aria-hidden="true" size={31} />
      </div>
      <div className={styles.grid2}>
        <div>
          <div
            className={styles.canvasGrid}
            style={{ gridTemplateColumns: "repeat(4, minmax(48px, 1fr))" }}
            aria-label="Gridworld"
          >
            {Array.from({ length: 16 }, (_, index) => {
              const row = Math.floor(index / 4);
              const column = index % 4;
              const isAgent = row === current[0] && column === current[1];
              const isGoal = row === 0 && column === 3;
              const isHazard = row === 2 && column === 2;
              const isPossible = highlighted.has(`${row}-${column}`);
              return (
                <div
                  className={styles.canvasCell}
                  key={index}
                  style={{
                    display: "grid",
                    placeItems: "center",
                    border: isPossible
                      ? "3px solid var(--accent)"
                      : "1px solid var(--line)",
                    background: isGoal
                      ? "#c9f1d7"
                      : isHazard
                        ? "#ffd7d3"
                        : isAgent
                          ? "var(--accent-soft)"
                          : "white",
                    fontWeight: 800,
                  }}
                >
                  {isAgent
                    ? "AGENT"
                    : isGoal
                      ? "+5"
                      : isHazard
                        ? "−4"
                        : VALUES[row]?.[column]?.toFixed(1)}
                </div>
              );
            })}
          </div>
          <p className={styles.status}>
            Numbers show estimated state values. A blocked move stays in the
            same cell.
          </p>
        </div>
        <div>
          <div className={styles.buttonRow}>
            {ACTIONS.map((action, index) => (
              <button
                type="button"
                aria-label={`Choose ${action.name}`}
                key={action.name}
                className={
                  actionIndex === index ? styles.buttonActive : styles.button
                }
                onClick={() => setActionIndex(index)}
              >
                {action.symbol} {action.name}
              </button>
            ))}
          </div>
          <label className={styles.control}>
            <span>Discount factor γ: {gamma.toFixed(2)}</span>
            <input
              aria-label="Gridworld discount factor"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={gamma}
              onChange={(event) => setGamma(Number(event.target.value))}
            />
          </label>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Actual move</th>
                  <th scope="col">Chance</th>
                  <th scope="col">Next state</th>
                  <th scope="col">Reward</th>
                </tr>
              </thead>
              <tbody>
                {outcomes.map((outcome, index) => (
                  <tr key={`${outcome.actualAction}-${index}`}>
                    <th scope="row">{ACTIONS[outcome.actualAction]?.name}</th>
                    <td>{(outcome.probability * 100).toFixed(0)}%</td>
                    <td>
                      ({outcome.next[0] + 1}, {outcome.next[1] + 1})
                    </td>
                    <td>{outcome.reward.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="expected immediate reward"
          value={expectedReward.toFixed(2)}
          detail="average over next-state branches"
        />
        <ProbabilityMetric
          label="Q estimate"
          value={qEstimate.toFixed(2)}
          detail="reward + discounted next value"
        />
      </div>
      <ProbabilityFormula
        label="One-step action value"
        formula={String.raw`\[Q(s,a)=\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+\gamma V(s')\right]=${qEstimate.toFixed(3)}\]`}
      >
        The policy chooses an action. The transition model describes the world’s
        response. Mixing those two distributions is a common conceptual error.
      </ProbabilityFormula>
    </div>
  );
}

function MarkovStateLab() {
  const [stateDesign, setStateDesign] = useState<
    "position" | "position-battery"
  >("position");
  const sufficient = stateDesign === "position-battery";
  return (
    <div className={styles.lab} data-testid="markov-state-lab">
      <div className={styles.labHeader}>
        <div>
          <p>State-design audit</p>
          <h3>The Markov property depends on what you put in “state”</h3>
        </div>
        <Bot aria-hidden="true" size={30} />
      </div>
      <p>
        A delivery robot’s chance of completing a long move depends on its
        battery. Which representation contains enough present information?
      </p>
      <div className={styles.buttonRow}>
        <button
          type="button"
          className={!sufficient ? styles.buttonActive : styles.button}
          onClick={() => setStateDesign("position")}
        >
          position only
        </button>
        <button
          type="button"
          className={sufficient ? styles.buttonActive : styles.button}
          onClick={() => setStateDesign("position-battery")}
        >
          position + battery
        </button>
      </div>
      <ProbabilityInsight
        title={
          sufficient
            ? "Now the present can screen off the past"
            : "The past still leaks information"
        }
        tone={sufficient ? "success" : "warning"}
      >
        <p>
          {sufficient
            ? "If position and battery capture everything relevant to the next transition, histories that reach the same state can share one transition model."
            : "How long the robot has travelled helps predict its battery, so two histories at the same position can have different futures. The proposed state is not Markov."}
        </p>
      </ProbabilityInsight>
      <ProbabilityFormula
        label="Markov property"
        formula={String.raw`\[P(S_{t+1}\mid S_t,A_t,S_{t-1},A_{t-1},\ldots)=P(S_{t+1}\mid S_t,A_t)\]`}
      >
        This is an assumption about a representation—not a claim that real
        systems have no history. A good state summarizes the relevant history.
      </ProbabilityFormula>
    </div>
  );
}

function ReturnAndTrajectoryLab() {
  const [gamma, setGamma] = useState(0.8);
  const [riskyFinalStep, setRiskyFinalStep] = useState(false);
  const rewards = [-0.1, -0.1, -0.1, 5];
  const stepProbabilities = riskyFinalStep
    ? [0.8, 0.8, 0.8, 0.35]
    : [0.8, 0.8, 0.8, 0.8];
  const result = discountedReturn(rewards, gamma);
  const pathProbability = trajectoryProbability(stepProbabilities);
  return (
    <div className={styles.lab} data-testid="return-trajectory-lab">
      <div className={styles.labHeader}>
        <div>
          <p>One possible future</p>
          <h3>
            Probability says how likely a path is; return says how valuable it
            is
          </h3>
        </div>
        <Footprints aria-hidden="true" size={31} />
      </div>
      <div className={styles.controls}>
        <label className={styles.control}>
          <span>Discount γ: {gamma.toFixed(2)}</span>
          <input
            aria-label="Trajectory discount factor"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={gamma}
            onChange={(event) => setGamma(Number(event.target.value))}
          />
        </label>
      </div>
      <div className={styles.buttonRow}>
        <button
          type="button"
          className={!riskyFinalStep ? styles.buttonActive : styles.button}
          onClick={() => setRiskyFinalStep(false)}
        >
          steady final move
        </button>
        <button
          type="button"
          className={riskyFinalStep ? styles.buttonActive : styles.button}
          onClick={() => setRiskyFinalStep(true)}
        >
          risky final move
        </button>
      </div>
      <div className={styles.sequenceStrip}>
        {rewards.map((reward, index) => (
          <div className={styles.sequenceItem} key={index}>
            <strong>step {index + 1}</strong>
            <span>p {stepProbabilities[index]?.toFixed(2)}</span>
            <span>
              r {reward > 0 ? "+" : ""}
              {reward}
            </span>
          </div>
        ))}
      </div>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="path probability"
          value={`${(pathProbability * 100).toFixed(1)}%`}
          detail="multiply transition chances"
        />
        <ProbabilityMetric
          label="path return"
          value={result.toFixed(3)}
          detail="discount and add rewards"
        />
      </div>
      <div className={styles.grid2}>
        <ProbabilityFormula
          label="Trajectory probability"
          formula={String.raw`\[P(\tau)=\prod_t \pi(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t)\]`}
        />
        <ProbabilityFormula
          label="Discounted return"
          formula={String.raw`\[G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots\]`}
        />
      </div>
    </div>
  );
}

function WaitingTimeLab() {
  const [runLength, setRunLength] = useState(2);
  const expectation = expectedFlipsForConsecutiveHeads(runLength);
  return (
    <div className={styles.lab} data-testid="waiting-time-lab">
      <div className={styles.labHeader}>
        <div>
          <p>First-step recursion</p>
          <h3>
            Stop listing infinite histories; define the states that matter
          </h3>
        </div>
        <Route aria-hidden="true" size={31} />
      </div>
      <label className={styles.control}>
        <span>Consecutive heads needed: {runLength}</span>
        <input
          aria-label="Consecutive heads needed"
          type="range"
          min="1"
          max="4"
          step="1"
          value={runLength}
          onChange={(event) => setRunLength(Number(event.target.value))}
        />
      </label>
      <div className={styles.metricGrid}>
        <ProbabilityMetric
          label="expected flips"
          value={String(expectation)}
          detail={`until ${"H".repeat(runLength)}`}
        />
        <ProbabilityMetric
          label="state count"
          value={String(runLength)}
          detail="current run lengths 0 through target − 1"
        />
      </div>
      {runLength === 1 ? (
        <ProbabilityFormula
          label="One state equation"
          formula={String.raw`\[E_0=1+\tfrac12(0)+\tfrac12E_0\quad\Rightarrow\quad E_0=2\]`}
        />
      ) : runLength === 2 ? (
        <ProbabilityFormula
          label="Two state equations"
          formula={String.raw`\[E_0=1+\tfrac12E_1+\tfrac12E_0,\qquad E_1=1+\tfrac12(0)+\tfrac12E_0\quad\Rightarrow\quad E_0=6\]`}
        />
      ) : (
        <ProbabilityFormula
          label="Solved recurrence"
          formula={String.raw`\[E_0=2^{k+1}-2=2^{${runLength + 1}}-2=${expectation}\]`}
        />
      )}
      <ProbabilityInsight title="The state remembers only what can affect the future">
        <p>
          After a tail, the current head run returns to zero. After a head, it
          advances by one. That compact memory is exactly the Markov-state idea
          used by Bellman equations.
        </p>
      </ProbabilityInsight>
    </div>
  );
}

function ExplorationLab() {
  const [epsilon, setEpsilon] = useState(20);
  const probabilities = epsilonGreedyDistribution(4, 1, epsilon / 100);
  return (
    <div className={styles.lab} data-testid="exploration-lab">
      <div className={styles.labHeader}>
        <div>
          <p>ε-greedy policy</p>
          <h3>Controlled randomness buys information</h3>
        </div>
        <Route aria-hidden="true" size={31} />
      </div>
      <label className={styles.control}>
        <span>Exploration ε: {epsilon}%</span>
        <input
          aria-label="Exploration epsilon"
          type="range"
          min="0"
          max="100"
          step="5"
          value={epsilon}
          onChange={(event) => setEpsilon(Number(event.target.value))}
        />
      </label>
      <div className={styles.bars}>
        {probabilities.map((probability, index) => (
          <div className={styles.barRow} key={ACTIONS[index]?.name}>
            <span>
              {ACTIONS[index]?.name}
              {index === 1 ? " · current best" : ""}
            </span>
            <div className={styles.barTrack}>
              <span
                className={styles.barFill}
                style={{ width: `${(probability * 100).toFixed(4)}%` }}
              />
            </div>
            <strong>{(probability * 100).toFixed(1)}%</strong>
          </div>
        ))}
      </div>
      <ProbabilityFormula
        label="ε-greedy distribution"
        formula={String.raw`\[\pi(a\mid s)=\begin{cases}1-\varepsilon+\varepsilon/4 & a=a^*\\ \varepsilon/4 & \text{otherwise}\end{cases}\]`}
      >
        Exploration is not wasted motion: early evidence can correct a mistaken
        value estimate. But persistent randomness also carries opportunity and
        safety costs, so exploration schedules and constraints matter.
      </ProbabilityFormula>
    </div>
  );
}

export default function CrashProbabilityL4LearningPage({ experience }: Props) {
  return (
    <ProbabilityCourse
      experience={experience}
      station="l4"
      kicker="Station L4 · Probability over time"
      headline="An action is a bet whose outcome becomes the next situation."
      introduction="Reinforcement learning joins probability to control. Policies randomize choices, transition models randomize consequences, rewards value what follows, and recursive expectations compress branching futures into reusable state values."
      heroVisual={<HeroDecisionWorld />}
    >
      <ProbabilitySection
        id="mdp"
        eyebrow="01 · State, action, transition, reward"
        title="Separate the agent’s choice from the world’s response."
        lead="A Markov decision process is a compact model of sequential uncertainty. The gridworld makes the two sources of randomness explicit: a policy chooses what to attempt, while transition probabilities determine where that attempt lands."
      >
        <GridworldLab />
        <ProbabilityCheck
          testId="l4-transition-check"
          title="Name the distribution"
          question="A robot chooses ‘up,’ then slips right with probability 0.1. Which object owns that 0.1?"
          options={[
            {
              label: "The policy",
              explanation:
                "The policy describes which action the agent chooses, not what the environment does afterward.",
            },
            {
              label: "The transition model",
              explanation:
                "It is the environment’s probability of a next state given current state and chosen action.",
            },
            {
              label: "The reward function",
              explanation:
                "Reward scores an outcome; it does not assign the probability of reaching it.",
            },
          ]}
          correctIndex={1}
        />
      </ProbabilitySection>

      <ProbabilitySection
        id="markov"
        eyebrow="02 · Make the present sufficient"
        title="Markov does not mean memoryless reality; it means adequate state."
        lead="The future may depend on history. The modeling move is to carry enough relevant history into the current state so earlier details add no predictive information about the next step."
      >
        <MarkovStateLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="return"
        eyebrow="03 · Paths, return, and value"
        title="A rare path can be valuable; a common path can be poor."
        lead="Return discounts and adds rewards along one realized path. A value function averages those returns over all future policy and transition branches."
      >
        <ReturnAndTrajectoryLab />
        <div className={styles.grid2} style={{ marginTop: 22 }}>
          <ProbabilityFormula
            label="State value"
            formula={String.raw`\[V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]\]`}
          >
            How good is it to be in state s and then follow policy π?
          </ProbabilityFormula>
          <ProbabilityFormula
            label="Action value"
            formula={String.raw`\[Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]\]`}
          >
            How good is choosing a now, before returning to policy π?
          </ProbabilityFormula>
        </div>
      </ProbabilitySection>

      <ProbabilitySection
        id="recursion"
        eyebrow="04 · First-step analysis"
        title="Condition on the next event and let the future call itself."
        lead="Waiting times appear to require infinitely many possible sequences. Recursive expectation replaces that list with one equation per relevant state—the same structure behind value iteration and Bellman equations."
      >
        <WaitingTimeLab />
      </ProbabilitySection>

      <ProbabilitySection
        id="exploration"
        eyebrow="05 · Exploration and exploitation"
        title="A policy can randomize on purpose."
        lead="Exploitation uses the best current estimate. Exploration tests alternatives that might be better. ε-greedy, softmax action selection, and entropy bonuses implement different kinds of controlled uncertainty."
      >
        <ExplorationLab />
        <ProbabilityInsight
          title="Partial observability changes the object"
          tone="warning"
        >
          <p>
            If the observation does not reveal the true state, the agent may
            need a belief distribution over hidden states or memory of previous
            observations. The Markov property can hold in belief state even when
            it fails for a single raw observation.
          </p>
        </ProbabilityInsight>
      </ProbabilitySection>

      <ProbabilityQuizLaunch
        experience={experience}
        recap={[
          "A policy distributes probability over actions; a transition model distributes probability over next states.",
          "A state is Markov when it contains enough present information to screen off earlier history for the next transition.",
          "Trajectory probabilities multiply policy and transition terms along a path.",
          "Return discounts realized rewards; value functions average future returns.",
          "First-step equations turn repeated or infinite random processes into finite recursive expectations.",
          "Exploration uses controlled randomness to improve knowledge while exploitation uses current estimates.",
        ]}
      />
    </ProbabilityCourse>
  );
}
