"use client";

import { useMemo, useState } from "react";
import MathText from "../../MathText";
import type { LearningExperience } from "../../../lib/learning";
import {
  CheckForUnderstanding,
  ConceptCard,
  FormulaBlock,
  LearningHero,
  MisconceptionCallout,
  ProcessSteps,
  QuizTransitionButton,
  RecapSection,
  WorkedExample,
} from "../LearningPrimitives";

type Props = {
  experience: LearningExperience;
};

type Direction = "up" | "right" | "down" | "left";

type Cell = {
  x: number;
  y: number;
};

type ActionConfig = {
  id: Direction;
  label: string;
  delta: Cell;
  slips: readonly [Direction, Direction];
};

type TransitionOutcome = {
  id: string;
  branchLabel: string;
  probability: number;
  cell: Cell;
  reward: number;
  nextValue: number;
  blocked: boolean;
};

const GRID_WIDTH = 4;
const GRID_HEIGHT = 3;
const CURRENT_CELL: Cell = { x: 1, y: 1 };
const GOAL_CELL: Cell = { x: 3, y: 0 };
const TRAP_CELL: Cell = { x: 2, y: 2 };
const WALL_KEYS = new Set(["0,1"]);

const ACTIONS: readonly ActionConfig[] = [
  {
    id: "up",
    label: "Up",
    delta: { x: 0, y: -1 },
    slips: ["left", "right"],
  },
  {
    id: "right",
    label: "Right",
    delta: { x: 1, y: 0 },
    slips: ["up", "down"],
  },
  {
    id: "down",
    label: "Down",
    delta: { x: 0, y: 1 },
    slips: ["right", "left"],
  },
  {
    id: "left",
    label: "Left",
    delta: { x: -1, y: 0 },
    slips: ["down", "up"],
  },
] as const;

const STATE_VALUES: Record<string, number> = {
  "0,0": 2,
  "1,0": 5,
  "2,0": 8,
  "3,0": 10,
  "1,1": 4,
  "2,1": 6,
  "3,1": 8,
  "0,2": 0,
  "1,2": 1,
  "2,2": -4,
  "3,2": 4,
};

const POLICY_ACTION_VALUES = [
  { id: "left", label: "left", value: 10 },
  { id: "right", label: "right", value: 6 },
  { id: "wait", label: "wait", value: 2 },
] as const;

function cellKey(cell: Cell): string {
  return `${cell.x},${cell.y}`;
}

function isSameCell(first: Cell, second: Cell): boolean {
  return first.x === second.x && first.y === second.y;
}

function getAction(actionId: Direction): ActionConfig {
  const action = ACTIONS.find((item) => item.id === actionId);
  if (!action) {
    throw new Error(`Unknown gridworld action: ${actionId}`);
  }
  return action;
}

function getStateValue(cell: Cell): number {
  return STATE_VALUES[cellKey(cell)] ?? 0;
}

function isBlocked(cell: Cell): boolean {
  return (
    cell.x < 0 ||
    cell.x >= GRID_WIDTH ||
    cell.y < 0 ||
    cell.y >= GRID_HEIGHT ||
    WALL_KEYS.has(cellKey(cell))
  );
}

function move(direction: Direction) {
  const action = getAction(direction);
  const nextCell = {
    x: CURRENT_CELL.x + action.delta.x,
    y: CURRENT_CELL.y + action.delta.y,
  };

  if (isBlocked(nextCell)) {
    return { cell: CURRENT_CELL, blocked: true };
  }

  return { cell: nextCell, blocked: false };
}

function getReward(cell: Cell, blocked: boolean): number {
  if (blocked) return -1;
  if (isSameCell(cell, GOAL_CELL)) return 10;
  if (isSameCell(cell, TRAP_CELL)) return -2;
  return -0.1;
}

function getTransitionOutcomes(actionId: Direction): TransitionOutcome[] {
  const action = getAction(actionId);
  const branches = [
    { label: `intended ${action.id}`, direction: action.id, probability: 0.8 },
    {
      label: `slip ${action.slips[0]}`,
      direction: action.slips[0],
      probability: 0.1,
    },
    {
      label: `slip ${action.slips[1]}`,
      direction: action.slips[1],
      probability: 0.1,
    },
  ] as const;

  return branches.map((branch) => {
    const result = move(branch.direction);
    const reward = getReward(result.cell, result.blocked);

    return {
      id: `${branch.label}-${cellKey(result.cell)}`,
      branchLabel: branch.label,
      probability: branch.probability,
      cell: result.cell,
      reward,
      nextValue: getStateValue(result.cell),
      blocked: result.blocked,
    };
  });
}

function describeCell(cell: Cell, blocked: boolean): string {
  if (blocked) return "same cell after wall";
  if (isSameCell(cell, GOAL_CELL)) return "goal cell";
  if (isSameCell(cell, TRAP_CELL)) return "trap cell";
  if (isSameCell(cell, CURRENT_CELL)) return "current cell";
  return `cell (${cell.x + 1}, ${cell.y + 1})`;
}

function formatNumber(value: number): string {
  const formatted = value.toFixed(2);
  return formatted === "-0.00" ? "0.00" : formatted;
}

function formatProbability(value: number): string {
  return `${(value * 100).toFixed(value * 100 >= 10 ? 0 : 1)}%`;
}

function RlLoopVisual() {
  const steps = [
    { label: "State", value: "S_t", body: "robot location" },
    { label: "Action", value: "A_t", body: "policy choice" },
    { label: "Reward", value: "R_{t+1}", body: "feedback" },
    { label: "Next state", value: "S_{t+1}", body: "new situation" },
  ] as const;

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-5">
      <div className="grid gap-3">
        {steps.map((step, index) => (
          <div key={step.label} className="flex items-center gap-3">
            <div className="flex min-h-20 flex-1 flex-col justify-center rounded-md border border-slate-700 bg-slate-950 px-4 py-3">
              <p className="text-xs font-semibold uppercase tracking-wide text-sky-300">
                {step.label}
              </p>
              <MathText
                text={String.raw`\(${step.value}\)`}
                className="mt-1 text-lg font-semibold text-slate-50"
              />
              <p className="text-xs text-slate-400">{step.body}</p>
            </div>
            {index < steps.length - 1 && (
              <span className="text-lg font-semibold text-emerald-300">
                -&gt;
              </span>
            )}
          </div>
        ))}
      </div>
      <p className="mt-4 text-sm leading-6 text-slate-300">
        The agent changes the data stream by choosing actions, so probability
        now describes possible futures, not only labels.
      </p>
    </div>
  );
}

function GridworldDecisionLab() {
  const [selectedAction, setSelectedAction] = useState<Direction>("right");
  const [gamma, setGamma] = useState(0.8);
  const outcomes = useMemo(
    () => getTransitionOutcomes(selectedAction),
    [selectedAction],
  );
  const expectedImmediateReward = outcomes.reduce(
    (sum, outcome) => sum + outcome.probability * outcome.reward,
    0,
  );
  const expectedNextValue = outcomes.reduce(
    (sum, outcome) => sum + outcome.probability * outcome.nextValue,
    0,
  );
  const expectedLookahead = outcomes.reduce(
    (sum, outcome) =>
      sum + outcome.probability * (outcome.reward + gamma * outcome.nextValue),
    0,
  );
  const outcomeProbabilityByCell = new Map<string, number>();

  for (const outcome of outcomes) {
    const key = cellKey(outcome.cell);
    outcomeProbabilityByCell.set(
      key,
      (outcomeProbabilityByCell.get(key) ?? 0) + outcome.probability,
    );
  }

  return (
    <section
      data-testid="gridworld-decision-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
            Interactive model
          </p>
          <h2 className="mt-2 text-xl font-semibold text-slate-50">
            Gridworld expected-return lab
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Choose an action from the current state. The environment usually
            follows the action but sometimes slips sideways, so the value of the
            action is a probability-weighted average over possible next states.
          </p>
        </div>
        <p
          role="status"
          data-testid="gridworld-summary"
          className="rounded-md border border-emerald-500/40 bg-emerald-950/30 px-3 py-2 text-sm font-semibold text-emerald-100"
        >
          Selected action: {selectedAction} / expected lookahead{" "}
          {formatNumber(expectedLookahead)}
        </p>
      </div>

      <div className="mt-5 grid gap-6 md:grid-cols-[0.95fr_1.05fr]">
        <div className="space-y-5">
          <div className="grid grid-cols-4 gap-2" aria-label="Gridworld board">
            {Array.from({ length: GRID_HEIGHT * GRID_WIDTH }, (_, index) => {
              const cell = {
                x: index % GRID_WIDTH,
                y: Math.floor(index / GRID_WIDTH),
              };
              const key = cellKey(cell);
              const isCurrent = isSameCell(cell, CURRENT_CELL);
              const isGoal = isSameCell(cell, GOAL_CELL);
              const isTrap = isSameCell(cell, TRAP_CELL);
              const isWall = WALL_KEYS.has(key);
              const outcomeProbability = outcomeProbabilityByCell.get(key);
              const isOutcome = outcomeProbability !== undefined;
              const cellClass = [
                "relative flex aspect-square min-h-[4.25rem] flex-col items-center justify-center rounded-md border p-2 text-center text-xs",
                isCurrent
                  ? "border-sky-300 bg-sky-950/60 text-sky-100"
                  : "border-slate-700 bg-slate-950 text-slate-300",
                isGoal ? "border-emerald-300 bg-emerald-950/50" : "",
                isTrap ? "border-amber-300 bg-amber-950/40" : "",
                isWall ? "border-slate-700 bg-slate-800 text-slate-500" : "",
                isOutcome ? "ring-2 ring-violet-300" : "",
              ].join(" ");

              return (
                <div key={key} className={cellClass}>
                  {isOutcome && (
                    <span className="absolute right-1 top-1 rounded bg-violet-300 px-1.5 py-0.5 text-[0.65rem] font-bold text-slate-950">
                      {formatProbability(outcomeProbability)}
                    </span>
                  )}
                  <span className="font-semibold">
                    {isWall
                      ? "Wall"
                      : isCurrent
                        ? "S_t"
                        : isGoal
                          ? "Goal"
                          : isTrap
                            ? "Trap"
                            : `(${cell.x + 1},${cell.y + 1})`}
                  </span>
                  {!isWall && (
                    <span className="mt-1 font-mono text-[0.68rem]">
                      V={formatNumber(getStateValue(cell))}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          <div>
            <p className="text-sm font-semibold text-slate-100">
              Choose action
            </p>
            <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {ACTIONS.map((action) => (
                <button
                  key={action.id}
                  type="button"
                  aria-pressed={selectedAction === action.id}
                  onClick={() => setSelectedAction(action.id)}
                  className={`rounded-md border px-3 py-2 text-sm font-semibold transition-colors ${
                    selectedAction === action.id
                      ? "border-sky-300 bg-sky-300 text-slate-950"
                      : "border-slate-700 bg-slate-950 text-slate-200 hover:border-slate-500"
                  }`}
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>

          <label className="block space-y-2">
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className="font-semibold text-slate-100">
                Discount factor gamma
              </span>
              <span className="font-mono text-slate-300">
                {formatNumber(gamma)}
              </span>
            </div>
            <input
              type="range"
              aria-label="Discount factor gamma"
              min="0"
              max="0.99"
              step="0.01"
              value={gamma}
              onChange={(event) => setGamma(Number(event.target.value))}
              className="w-full accent-emerald-400"
            />
          </label>
        </div>

        <div className="space-y-4">
          <MathText
            text={String.raw`\[\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+\gamma V(s')\right]\]`}
            className="overflow-x-auto rounded-md border border-slate-700 bg-slate-950 px-4 py-3 text-slate-100"
          />

          <div className="grid gap-3">
            {outcomes.map((outcome) => {
              const contribution =
                outcome.probability *
                (outcome.reward + gamma * outcome.nextValue);

              return (
                <div
                  key={outcome.id}
                  className="rounded-md border border-slate-800 bg-slate-950 p-3"
                >
                  <div className="flex flex-wrap items-start justify-between gap-3 text-sm">
                    <div>
                      <p className="font-semibold text-slate-100">
                        {outcome.branchLabel}
                      </p>
                      <p className="mt-1 text-slate-400">
                        {describeCell(outcome.cell, outcome.blocked)}
                      </p>
                    </div>
                    <p className="font-mono text-slate-300">
                      p={formatProbability(outcome.probability)}
                    </p>
                  </div>
                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full bg-violet-300"
                      style={{
                        width: `${Math.max(4, outcome.probability * 100)}%`,
                      }}
                    />
                  </div>
                  <p className="mt-3 text-xs leading-5 text-slate-300">
                    reward {formatNumber(outcome.reward)} + gamma value{" "}
                    {formatNumber(gamma * outcome.nextValue)} contributes{" "}
                    {formatNumber(contribution)}.
                  </p>
                </div>
              );
            })}
          </div>

          <div className="grid gap-2 rounded-md border border-slate-700 bg-slate-950 p-3 text-sm text-slate-200">
            <p>
              Expected immediate reward:{" "}
              <span className="font-mono">
                {formatNumber(expectedImmediateReward)}
              </span>
            </p>
            <p>
              Expected next-state value:{" "}
              <span className="font-mono">
                {formatNumber(expectedNextValue)}
              </span>
            </p>
            <p>
              Expected one-step lookahead:{" "}
              <span className="font-mono">
                {formatNumber(expectedLookahead)}
              </span>
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function ExplorationPolicyLab() {
  const [epsilon, setEpsilon] = useState(0.1);
  const greedyAction = POLICY_ACTION_VALUES.reduce((best, action) =>
    action.value > best.value ? action : best,
  );
  const actionCount = POLICY_ACTION_VALUES.length;
  const policy = POLICY_ACTION_VALUES.map((action) => {
    const explorationProbability = epsilon / actionCount;
    const exploitationProbability =
      action.id === greedyAction.id ? 1 - epsilon : 0;

    return {
      ...action,
      probability: explorationProbability + exploitationProbability,
    };
  });
  const expectedSelectedValue = policy.reduce(
    (sum, action) => sum + action.probability * action.value,
    0,
  );

  return (
    <section
      data-testid="exploration-policy-lab"
      className="rounded-lg border border-slate-800 bg-slate-900 p-5"
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-amber-300">
            Exploration
          </p>
          <h2 className="mt-2 text-xl font-semibold text-slate-50">
            Epsilon-greedy policy lab
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Pure exploitation always chooses the largest current Q-value.
            Epsilon-greedy keeps most probability on that action while reserving
            some probability for trying alternatives.
          </p>
        </div>
        <p
          role="status"
          data-testid="exploration-summary"
          className="rounded-md border border-amber-500/40 bg-amber-950/30 px-3 py-2 text-sm font-semibold text-amber-100"
        >
          Expected selected Q-value: {formatNumber(expectedSelectedValue)}
        </p>
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-[0.8fr_1.2fr]">
        <label className="block space-y-2">
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="font-semibold text-slate-100">
              Exploration rate epsilon
            </span>
            <span className="font-mono text-slate-300">
              {formatProbability(epsilon)}
            </span>
          </div>
          <input
            type="range"
            aria-label="Exploration rate epsilon"
            min="0"
            max="0.5"
            step="0.05"
            value={epsilon}
            onChange={(event) => setEpsilon(Number(event.target.value))}
            className="w-full accent-amber-400"
          />
          <p className="text-xs leading-5 text-slate-400">
            With probability 1 - epsilon, choose the best-known action. With
            probability epsilon, sample uniformly from all actions.
          </p>
        </label>

        <div className="grid gap-3">
          {policy.map((action) => (
            <div
              key={action.id}
              className="rounded-md border border-slate-800 bg-slate-950 p-3"
            >
              <div className="flex items-center justify-between gap-3 text-sm">
                <span className="font-semibold text-slate-100">
                  {action.label}
                </span>
                <span className="font-mono text-slate-300">
                  Q={formatNumber(action.value)} / pi=
                  {formatProbability(action.probability)}
                </span>
              </div>
              <div className="mt-2 h-3 overflow-hidden rounded-full bg-slate-800">
                <div
                  className="h-full rounded-full bg-amber-300"
                  style={{
                    width: `${Math.max(3, action.probability * 100)}%`,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function CrashProbabilityL4LearningPage({ experience }: Props) {
  return (
    <main className="bg-slate-950 text-slate-50">
      <LearningHero
        eyebrow="Crash Course Probability L4"
        title="Choose actions by averaging possible futures"
        summary="Reinforcement learning extends probability from prediction into action: a policy chooses, the environment responds stochastically, and the objective is expected future reward."
        meta={`${experience.durationMinutes} min interactive prep / ${experience.level}`}
        outcomes={experience.outcomes}
        visual={<RlLoopVisual />}
      />

      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-4 py-10 md:py-12">
        <ProcessSteps
          title="The probabilistic decision loop"
          steps={[
            {
              title: "Observe a state",
              body: "The state is the information the agent uses now. It should contain the relevant past when the problem is Markov.",
            },
            {
              title: "Sample or choose an action",
              body: "A policy maps the state to an action or to a probability distribution over actions.",
            },
            {
              title: "Average future rewards",
              body: "Transitions and rewards can be uncertain, so the agent compares expected return across possible futures.",
            },
          ]}
        />

        <section className="grid gap-4 md:grid-cols-4">
          <ConceptCard title="State" label="Situation">
            <p>
              The current information available to the agent, such as the robot
              cell, a chess board, or an LLM prompt plus context.
            </p>
          </ConceptCard>
          <ConceptCard title="Action" label="Choice">
            <p>
              The decision the agent controls: move, recommend, choose a token,
              or select a response strategy.
            </p>
          </ConceptCard>
          <ConceptCard title="Transition" label="World response">
            <p>
              The environment maps a state-action pair to possible next states,
              often with probabilities rather than guarantees.
            </p>
          </ConceptCard>
          <ConceptCard title="Reward" label="Feedback">
            <p>
              The scalar signal that says how useful the consequence was, from a
              step cost to a preference-model score.
            </p>
          </ConceptCard>
        </section>

        <FormulaBlock
          title="Two conditional distributions drive the loop"
          formula={String.raw`\[P(s'\mid s,a)\quad\text{and}\quad \pi(a\mid s)\]`}
          explanation="The environment distribution describes what could happen after an action. The policy distribution describes how the agent chooses actions from the state."
        />

        <GridworldDecisionLab />

        <CheckForUnderstanding
          testId="markov-check"
          title="Check: what makes a state Markov?"
          question="A chatbot state contains only the latest user message, but earlier turns determine what answer is appropriate. What is the right diagnosis?"
          correctIndex={1}
          options={[
            {
              label:
                "It is Markov because the latest message is the current observation.",
              explanation:
                "The current observation is not automatically a Markov state. It may leave out information needed to predict good next responses.",
            },
            {
              label:
                "It is probably not Markov unless the state includes relevant history or memory.",
              explanation:
                "The Markov property depends on whether the state contains the relevant past. Adding conversation history or memory can make the state closer to sufficient.",
            },
            {
              label: "It becomes Markov as soon as the policy is stochastic.",
              explanation:
                "Randomizing the policy does not restore missing state information. The representation still lacks relevant history.",
            },
          ]}
        />

        <FormulaBlock
          title="Return discounts future rewards"
          formula={String.raw`\[G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots\]`}
          explanation="A larger discount factor keeps later rewards influential. A smaller discount factor makes the agent more short-sighted."
        />

        <WorkedExample
          title="Worked example: same rewards, different patience"
          setup="The next rewards are 1, 2, and 10. Compare two discount factors."
          steps={[
            "With gamma = 0.5, return is 1 + 0.5*2 + 0.5^2*10 = 4.5.",
            "With gamma = 0.9, return is 1 + 0.9*2 + 0.9^2*10 = 10.9.",
            "The delayed reward is the same in both cases, but a high gamma preserves much more of its value.",
          ]}
        />

        <section className="grid gap-4 md:grid-cols-2">
          <ConceptCard title="State value" label="How good is this state?">
            <MathText
              text={String.raw`\[V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]\]`}
              className="overflow-x-auto rounded-md bg-slate-950 px-3 py-2 text-slate-100"
            />
            <p>
              The expected return from a state if the agent follows policy pi.
            </p>
          </ConceptCard>
          <ConceptCard title="Action value" label="How good is this action?">
            <MathText
              text={String.raw`\[Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]\]`}
              className="overflow-x-auto rounded-md bg-slate-950 px-3 py-2 text-slate-100"
            />
            <p>
              The expected return after taking action a in state s and then
              continuing with policy pi.
            </p>
          </ConceptCard>
        </section>

        <MisconceptionCallout
          misconception="The best action is the one with the biggest immediate reward."
          correction="RL value is about expected future return. A move with reward 0 can be better than a +1 coin if it leads to the goal, and a short-term gain can be bad if it creates a dead end."
        />

        <ExplorationPolicyLab />

        <CheckForUnderstanding
          title="Check: policy versus environment"
          question="Which distribution is controlled by the agent rather than by the environment?"
          correctIndex={0}
          options={[
            {
              label: "The policy distribution pi(a | s).",
              explanation:
                "The policy is the agent's behavior rule. It assigns probabilities to actions given the current state.",
            },
            {
              label: "The transition distribution P(s' | s, a).",
              explanation:
                "That distribution describes the environment's response after the agent acts.",
            },
            {
              label: "The reward-transition distribution P(r, s' | s, a).",
              explanation:
                "That is also part of the environment response, not the agent's action-selection rule.",
            },
          ]}
        />

        <FormulaBlock
          title="The RL objective is an expectation over futures"
          formula={String.raw`\[\max_\pi \mathbb{E}_\pi[G_t]\]`}
          explanation="The policy changes which trajectories become likely. Learning searches for a policy that makes high-return futures more probable under uncertainty."
        />

        <RecapSection
          title="Before you start the MCQs"
          items={[
            "RL changes the problem from predicting labels to choosing actions over time.",
            "Transition probabilities P(s' | s, a) describe uncertain environment consequences.",
            "A Markov state should contain the relevant history for predicting what comes next.",
            "A policy pi(a | s) can be deterministic or stochastic.",
            "Discounted return adds future rewards with powers of gamma.",
            "Value functions estimate expected future return, not only immediate reward.",
            "Exploration uses controlled randomness so the agent can learn about uncertain actions.",
            "RLHF uses reward signals to shape a language model's output distribution beyond pure imitation.",
          ]}
        />

        <section className="flex flex-col gap-4 rounded-lg border border-emerald-500/40 bg-emerald-950/20 p-5 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-emerald-100">
              Ready for the Probability L4 questions
            </h2>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Practice the same concepts with MCQs on states, actions,
              transition probabilities, Markov assumptions, policies, expected
              return, value functions, and exploration.
            </p>
          </div>
          <QuizTransitionButton sourceId={experience.sourceId} />
        </section>

        <section className="rounded-lg border border-slate-800 bg-slate-900 p-5">
          <h2 className="text-lg font-semibold text-slate-50">
            Compact formula board
          </h2>
          <div className="mt-4 space-y-3 text-slate-200">
            <MathText
              text={String.raw`\[S_t\rightarrow A_t\rightarrow R_{t+1},S_{t+1}\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[P(S_{t+1}\mid S_t,A_t,\text{history})=P(S_{t+1}\mid S_t,A_t)\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[\pi(a\mid s),\quad G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}\]`}
              className="overflow-x-auto"
            />
            <MathText
              text={String.raw`\[V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s],\quad Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]\]`}
              className="overflow-x-auto"
            />
          </div>
        </section>
      </div>
    </main>
  );
}
