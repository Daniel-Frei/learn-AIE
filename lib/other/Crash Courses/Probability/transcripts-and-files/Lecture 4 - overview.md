# Lecture 4 — Probability Over Time: Reinforcement Learning

**Theme:** Reinforcement learning is probability plus decisions over time.

**Duration:** 60 minutes
**Level:** After Lectures 1–3
**Style:** Mathematical but applied
**Core AI connection:** RL extends probability from prediction into action: the agent must choose actions under uncertainty to maximize expected future reward.

---

# Lecture 4 Overview

## Central Message

In supervised learning, the model usually predicts an output from an input:

[
x \rightarrow y
]

In reinforcement learning, the model acts in an environment:

[
S_t \rightarrow A_t \rightarrow R_{t+1}, S_{t+1}
]

The agent does not merely predict. It chooses actions, receives rewards, and changes what happens next.

The key shift is:

[
\text{prediction}
\quad \rightarrow \quad
\text{decision-making over time}
]

In Lecture 3, training meant:

[
\min -\log P(y \mid x)
]

In Lecture 4, learning means:

[
\max \mathbb{E}[\text{future reward}]
]

So the central probabilistic idea is expectation over possible futures.

---

# Lecture Structure

| Part |                           Topic |   Time |
| ---- | ------------------------------: | -----: |
| 1    |       From prediction to action |  7 min |
| 2    |        States, actions, rewards |  8 min |
| 3    |        Transition probabilities |  8 min |
| 4    |             The Markov property |  7 min |
| 5    |                        Policies |  8 min |
| 6    |                 Expected return | 10 min |
| 7    |                 Value functions |  8 min |
| 8    |     Exploration vs exploitation |  3 min |
| 9    | Summary and bridge to Lecture 5 |  1 min |

---

# Learning Goals

By the end of the lecture, students should understand:

- how RL differs from supervised learning,
- what states, actions, rewards, and next states are,
- what transition probabilities mean,
- what the Markov property says,
- what a policy is,
- why policies can be stochastic,
- what expected return means,
- what the discount factor does,
- what value functions estimate,
- why exploration requires randomness,
- how RL connects to deep learning and LLM alignment/RLHF.

---

# Running Example for the Lecture

Use one simple example throughout: a robot in a gridworld.

The robot is in a small grid. It can choose:

[
A = {\text{up}, \text{down}, \text{left}, \text{right}}
]

It wants to reach a goal.

- Reaching the goal gives reward (+10).
- Hitting a wall gives reward (-1).
- Each normal step gives reward (-0.1).
- Actions are uncertain: if the robot chooses “up,” it usually moves up, but sometimes slips sideways.

This example is useful because it naturally illustrates:

- state,
- action,
- reward,
- next state,
- stochastic transitions,
- policy,
- value,
- exploration.

---

# Part 1 — From Prediction to Action

**Time:** 7 minutes

## 1.1 Recap from Lecture 3

In supervised learning, we have examples:

[
(x_1,y_1), (x_2,y_2), ..., (x_n,y_n)
]

The model learns:

[
P(y \mid x)
]

Example:

[
P(\text{cat} \mid \text{image})
]

or:

[
P(\text{next token} \mid \text{previous tokens})
]

The model predicts, gets compared to the correct answer, and receives a loss.

---

## 1.2 The RL Difference

In reinforcement learning, the agent is not just predicting a label.

It acts.

The action changes the future.

Basic loop:

[
S_t \rightarrow A_t \rightarrow R_{t+1}, S_{t+1}
]

In words:

1. The agent observes the current state.
2. It chooses an action.
3. The environment gives a reward.
4. The environment moves to a new state.
5. The agent repeats.

This creates a sequence:

[
S_0, A_0, R_1, S_1, A_1, R_2, S_2, ...
]

The important difference:

> In supervised learning, the data is usually given.
> In reinforcement learning, the agent’s actions influence the data it will see next.

---

## 1.3 Why Probability Is Needed

RL needs probability because:

- actions may have uncertain effects,
- future states are uncertain,
- rewards may be noisy,
- the agent may use random exploration,
- the agent maximizes expected future reward, not guaranteed reward.

Example:

The robot chooses “up.”

Possible outcomes:

| Outcome     | Probability |
| ----------- | ----------: |
| moves up    |         0.8 |
| slips left  |         0.1 |
| slips right |         0.1 |

So the same action can lead to different next states.

That is why RL is probabilistic.

---

## 1.4 Core Comparison

| Supervised Learning                      | Reinforcement Learning            |
| ---------------------------------------- | --------------------------------- |
| Predict output                           | Choose action                     |
| Data is usually fixed                    | Data depends on agent behavior    |
| Feedback is usually immediate label/loss | Feedback may be delayed reward    |
| Goal: minimize prediction loss           | Goal: maximize expected return    |
| Example: classify image                  | Example: play game, control robot |

Core sentence:

> Reinforcement learning is about choosing actions whose consequences unfold over time.

---

## 1.5 Mini-Exercise

Ask students:

Which of these are supervised learning problems, and which are RL problems?

1. Predict whether an email is spam.
2. Choose moves in chess to eventually win.
3. Predict the next word in a sentence.
4. Recommend videos and learn from user reactions.
5. Control a robot arm to pick up an object.

Expected answers:

1. Supervised learning.
2. RL.
3. Supervised/self-supervised next-token prediction.
4. Can be framed as RL or contextual bandit/recommendation.
5. RL/control.

Teaching point:

> RL is especially natural when actions influence future observations or rewards.

---

# Part 2 — States, Actions, and Rewards

**Time:** 8 minutes

## 2.1 State

A **state** describes the current situation.

Notation:

[
s
]

or at time (t):

[
S_t
]

Examples:

| Domain                    | State                                        |
| ------------------------- | -------------------------------------------- |
| Gridworld robot           | robot’s position                             |
| Chess                     | board position                               |
| Video game                | screen/frame or game variables               |
| Recommendation            | user context and history                     |
| Chatbot                   | conversation state                           |
| Clinical decision support | patient state, history, current measurements |

In the gridworld:

[
S_t = \text{robot's current cell}
]

---

## 2.2 Action

An **action** is what the agent chooses to do.

Notation:

[
a
]

or:

[
A_t
]

Examples:

| Domain          | Action                             |
| --------------- | ---------------------------------- |
| Gridworld robot | up, down, left, right              |
| Chess           | legal move                         |
| Recommendation  | show item                          |
| Chatbot         | produce response / choose strategy |
| RLHF            | choose or favor one response       |
| Robot arm       | move joint angles                  |

In the gridworld:

[
A_t \in {\text{up}, \text{down}, \text{left}, \text{right}}
]

---

## 2.3 Reward

A **reward** is the feedback signal.

Notation:

[
r
]

or:

[
R_{t+1}
]

The reward tells the agent how good or bad the consequence was.

Examples:

| Domain         | Reward                                               |
| -------------- | ---------------------------------------------------- |
| Game           | score, win/loss                                      |
| Robot          | distance to goal, success/failure                    |
| Recommendation | click, watch time, satisfaction                      |
| Chatbot        | human preference score                               |
| RLHF           | reward model score                                   |
| Medical AI     | improved outcome, but usually used carefully/offline |

In the gridworld:

- reaching goal: (+10),
- hitting wall: (-1),
- normal step: (-0.1).

---

## 2.4 Next State

After the action, the environment moves to a next state:

[
S_{t+1}
]

Example:

If the robot is in cell ((2,2)) and moves up, it may end in:

[
(2,3)
]

But if the environment is stochastic, it may slip and end in another nearby cell.

---

## 2.5 The RL Loop

Write the loop clearly:

[
S_t \xrightarrow{\text{agent chooses } A_t} R_{t+1}, S_{t+1}
]

Then repeat:

[
S_{t+1} \xrightarrow{\text{agent chooses } A_{t+1}} R_{t+2}, S_{t+2}
]

This is why RL is sequential.

---

## 2.6 Trajectory

A full sequence of interaction is called a trajectory or episode.

[
\tau =
(S_0,A_0,R_1,S_1,A_1,R_2,S_2,...)
]

Example:

[
\text{start} \rightarrow \text{move right} \rightarrow -0.1 \rightarrow \text{new cell} \rightarrow \text{move up} \rightarrow +10
]

The agent learns from trajectories.

---

## 2.7 AI Examples

### Game Playing

- state: board position,
- action: move,
- reward: win/loss or score.

### Robotics

- state: sensor readings and position,
- action: motor command,
- reward: task success, energy cost, collision penalty.

### Recommendation Systems

- state: user profile and context,
- action: item shown,
- reward: click, purchase, watch time, satisfaction.

### RLHF for LLMs

- state/context: prompt and generated partial answer,
- action: token or response choice,
- reward: human preference model or reward model score.

Important caveat:

> RLHF is not the same as ordinary game RL, but it uses RL-style optimization with reward signals to shape model behavior.

---

## 2.8 Mini-Exercise

For each example, identify state, action, and reward.

1. A robot vacuum cleaning a room.
2. A chess-playing AI.
3. A recommender system suggesting YouTube videos.
4. A chatbot trained with human preference feedback.

Expected answers:

1. State: room/map/battery/position; action: move/turn/clean; reward: clean area, battery cost, collision penalty.
2. State: board; action: move; reward: win/loss/material/position score.
3. State: user/context/history; action: recommend video; reward: click/watch/satisfaction.
4. State: prompt/conversation; action: answer/token/response; reward: human preference or reward model score.

---

# Part 3 — Transition Probabilities

**Time:** 8 minutes

## 3.1 The Environment Is Often Stochastic

In many RL problems, actions do not have guaranteed outcomes.

The same action in the same state can lead to different next states.

This is described by transition probabilities:

[
P(s' \mid s,a)
]

Meaning:

> Given current state (s) and action (a), what is the probability of ending up in next state (s')?

---

## 3.2 Gridworld Example

The robot is in cell (s) and chooses action:

[
a = \text{up}
]

Possible next states:

| Next state (s')   | Probability |
| ----------------- | ----------: |
| cell above        |         0.8 |
| cell to the left  |         0.1 |
| cell to the right |         0.1 |

So:

[
P(\text{above} \mid s, \text{up}) = 0.8
]

[
P(\text{left} \mid s, \text{up}) = 0.1
]

[
P(\text{right} \mid s, \text{up}) = 0.1
]

The probabilities sum to 1:

[
0.8 + 0.1 + 0.1 = 1
]

This is just a conditional probability distribution over next states.

---

## 3.3 Deterministic vs Stochastic Environments

### Deterministic environment

The same state-action pair always gives the same next state.

[
P(s' \mid s,a) = 1
]

for one next state.

Example:

A simple board game with no randomness.

### Stochastic environment

The same state-action pair can lead to different next states.

Example:

- robot slipping,
- random opponent behavior,
- noisy sensors,
- user behavior in recommendation systems,
- medical treatment responses,
- stock market behavior.

RL often assumes uncertainty because the real world is uncertain.

---

## 3.4 Transition Probabilities Are Conditional Probabilities

Connect to Lecture 2:

[
P(s' \mid s,a)
]

has the same structure as:

[
P(y \mid x)
]

But now:

- input/context: current state and action,
- output: next state.

So RL uses conditional probability over time.

---

## 3.5 Rewards Can Also Be Stochastic

Sometimes the reward is also uncertain.

We might model:

[
P(r \mid s,a)
]

or:

[
P(r, s' \mid s,a)
]

Meaning:

> Given state and action, what reward and next state might occur?

Example:

A recommendation system shows a video.

The user may:

| User response      | Reward | Probability |
| ------------------ | -----: | ----------: |
| clicks and watches |     +5 |         0.2 |
| clicks and leaves  |     +1 |         0.1 |
| ignores            |      0 |         0.7 |

The reward depends probabilistically on user behavior.

---

## 3.6 Expected Immediate Reward

If rewards are stochastic, we can compute expected reward.

Example:

| Reward | Probability |
| -----: | ----------: |
|    +10 |         0.2 |
|      0 |         0.7 |
|     -1 |         0.1 |

Expected reward:

[
\mathbb{E}[R \mid s,a]
======================

10 \cdot 0.2 + 0 \cdot 0.7 + (-1) \cdot 0.1
]

[
= 2 - 0.1 = 1.9
]

This is not yet full RL return, but it prepares students for expected return.

---

## 3.7 Mini-Exercise

A robot chooses action “forward.”

| Outcome       | Reward | Probability |
| ------------- | -----: | ----------: |
| moves forward |     +2 |         0.7 |
| slips         |     -1 |         0.2 |
| stays still   |      0 |         0.1 |

Questions:

1. What is (P(\text{moves forward} \mid s, \text{forward}))?
2. What is (P(\text{slips} \mid s, \text{forward}))?
3. What is the expected immediate reward?
4. Why is this a conditional probability distribution?

Expected answers:

1. (0.7)
2. (0.2)
3. (2 \cdot 0.7 + (-1)\cdot 0.2 + 0 \cdot 0.1 = 1.2)
4. Because probabilities are over outcomes given the current state and action.

---

# Part 4 — The Markov Property

**Time:** 7 minutes

## 4.1 Motivation

In a sequential problem, the future could depend on the entire past:

[
S_0,A_0,S_1,A_1,...,S_t,A_t
]

That is complicated.

The Markov property simplifies this.

---

## 4.2 The Markov Property

The Markov assumption says:

> The future depends on the current state, not the full past history.

Formally:

[
P(S*{t+1} \mid S_t,A_t,S*{t-1},A\_{t-1},...,S_0,A_0)
===================================================

P(S\_{t+1} \mid S_t,A_t)
]

In words:

> Once you know the current state and action, the earlier history does not add extra information about the next state.

---

## 4.3 Intuition

The Markov property does **not** mean history is irrelevant in a human sense.

It means:

> The current state should contain all relevant information from the past.

Example:

In chess, the board position mostly summarizes what matters for the next move.

In a robot vacuum, the current map, position, battery, and obstacle information may summarize what matters.

In a medical setting, “current state” may need to include history, lab values, medications, age, prior diagnoses, etc.

So the Markov property depends on how we define the state.

---

## 4.4 Markov Decision Process

An RL problem is often modeled as a Markov Decision Process, or MDP.

An MDP contains:

[
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
]

Where:

| Symbol                  | Meaning                  |
| ----------------------- | ------------------------ |
| (\mathcal{S})           | set of states            |
| (\mathcal{A})           | set of actions           |
| (P(s' \mid s,a))        | transition probabilities |
| (R(s,a)) or (R(s,a,s')) | rewards                  |
| (\gamma)                | discount factor          |

Students do not need to memorize formal definitions deeply, but they should recognize the pieces.

---

## 4.5 When the Markov Assumption Fails

Sometimes the current observation does not contain all relevant information.

Example:

A robot sees a wall but does not know where it is on the map.

A medical AI sees current symptoms but not patient history.

A chatbot sees only the latest message but not earlier conversation.

Then the problem may be **partially observable**.

The agent may need memory, belief states, or context.

This connects to transformers:

> Transformers use context/history because many problems are not Markovian if the current token alone is treated as the state.

---

## 4.6 Transformer Contrast

In language modeling, the probability of the next token is:

[
P(X_t \mid X_1, X_2, ..., X_{t-1})
]

This uses the whole previous context.

In a Markov model, one would want something like:

[
P(S_{t+1} \mid S_t,A_t)
]

The bridge:

- In RL, we try to define (S_t) so it contains enough information.
- In transformers, the context window helps construct a rich representation of relevant history.

Useful teaching point:

> The Markov property is partly a modeling choice: it depends on whether the state representation is rich enough.

---

## 4.7 Mini-Exercise

Ask:

Is the following state representation likely Markov?

1. Chess state = full board position.
2. Chess state = only whose turn it is.
3. Chatbot state = only latest user message.
4. Chatbot state = full conversation history plus relevant memory.
5. Robot state = current camera image only in a partially hidden maze.

Expected answers:

1. Mostly yes, though some rules require history.
2. No.
3. Usually no.
4. More likely.
5. Often no, because hidden information may matter.

---

# Part 5 — Policies

**Time:** 8 minutes

## 5.1 What Is a Policy?

A policy tells the agent how to act.

Notation:

[
\pi
]

A policy maps states to actions or to probabilities over actions.

---

## 5.2 Deterministic Policy

A deterministic policy chooses one action for each state.

[
a = \pi(s)
]

Example:

If the robot is in this cell, always move right.

| State | Action |
| ----- | ------ |
| (s_1) | right  |
| (s_2) | up     |
| (s_3) | left   |

This is simple but not always ideal.

---

## 5.3 Stochastic Policy

A stochastic policy gives a probability distribution over actions.

[
\pi(a \mid s)
]

Meaning:

> Given state (s), what is the probability of choosing action (a)?

Example:

| Action | Probability |
| ------ | ----------: |
| up     |        0.70 |
| right  |        0.20 |
| left   |        0.05 |
| down   |        0.05 |

The agent will usually move up, but sometimes it explores other actions.

---

## 5.4 Policy as Conditional Probability

This connects directly to Lecture 2:

[
\pi(a \mid s)
]

has the same form as:

[
P(y \mid x)
]

But now:

- input (x) is state (s),
- output (y) is action (a).

So a policy network is like a classifier over actions.

---

## 5.5 Policy Network

A neural network policy can work like this:

[
s
\rightarrow
\text{neural network}
\rightarrow
\text{action logits}
\rightarrow
\text{softmax}
\rightarrow
\pi(a \mid s)
]

Example:

| Action | Logit | Probability |
| ------ | ----: | ----------: |
| up     |   2.0 |        0.66 |
| right  |   1.0 |        0.24 |
| left   |   0.0 |        0.09 |
| down   |  -1.0 |        0.03 |

This connects directly to Lecture 3.

---

## 5.6 LLM Connection

An LLM also outputs a probability distribution, but over tokens:

[
P(\text{next token} \mid \text{context})
]

An RL policy outputs a probability distribution over actions:

[
\pi(\text{action} \mid \text{state})
]

The analogy:

| LLM                     | RL            |
| ----------------------- | ------------- |
| context                 | state         |
| token                   | action        |
| next-token distribution | policy        |
| sample token            | choose action |
| generated text          | trajectory    |

This analogy is not perfect, but it helps students understand why RL and LLMs can be combined.

---

## 5.7 RLHF Connection

In RLHF, a language model’s outputs are adjusted using reward signals.

Simplified picture:

1. The model generates responses.
2. Humans or preference data compare responses.
3. A reward model is trained to score responses.
4. The language model is optimized to produce responses with higher reward.

The model is still producing token probabilities, but now training is influenced by a reward signal, not just next-token prediction.

Important:

> Supervised next-token training teaches the model to imitate data. RLHF tries to shape the model’s behavior according to preferences or goals.

---

## 5.8 Mini-Exercise

A policy in state (s) gives:

| Action | Probability |
| ------ | ----------: |
| left   |         0.6 |
| right  |         0.3 |
| wait   |         0.1 |

Questions:

1. Is this deterministic or stochastic?
2. What is (\pi(\text{left} \mid s))?
3. Which action is most likely?
4. Could the agent still choose “wait”?

Expected answers:

1. Stochastic.
2. (0.6)
3. Left.
4. Yes, with probability (0.1).

---

# Part 6 — Expected Return

**Time:** 10 minutes

## 6.1 Immediate Reward Is Not Enough

In RL, the best action is not always the one with the best immediate reward.

Example:

A robot can choose:

| Action           | Immediate Reward | Long-term Consequence |
| ---------------- | ---------------: | --------------------- |
| grab coin        |               +1 | dead end              |
| move toward goal |                0 | later reaches +10     |
| avoid obstacle   |             -0.1 | safer future          |

The agent needs to care about future rewards.

---

## 6.2 Return

The return is the cumulative future reward from time (t).

[
G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots
]

But often we discount future rewards.

Discounted return:

[
G*t =
R*{t+1}

- \gamma R\_{t+2}
- \gamma^2 R\_{t+3}
- \gamma^3 R\_{t+4}
- \cdots
  ]

Here:

[
0 \leq \gamma \leq 1
]

(\gamma) is the discount factor.

---

## 6.3 What the Discount Factor Does

The discount factor controls how much the agent cares about future rewards.

### Low (\gamma)

Example:

[
\gamma = 0.1
]

The agent mostly cares about immediate reward.

This creates a short-sighted agent.

### High (\gamma)

Example:

[
\gamma = 0.99
]

The agent cares a lot about future rewards.

This creates a patient agent.

### (\gamma = 0)

Only immediate reward matters:

[
G_t = R_{t+1}
]

### (\gamma \approx 1)

Future rewards matter almost as much as immediate rewards.

---

## 6.4 Numerical Example

Suppose:

[
R_{t+1} = 1
]

[
R_{t+2} = 2
]

[
R_{t+3} = 10
]

If:

[
\gamma = 0.5
]

then:

[
G_t = 1 + 0.5 \cdot 2 + 0.5^2 \cdot 10
]

[
G_t = 1 + 1 + 2.5 = 4.5
]

If:

[
\gamma = 0.9
]

then:

[
G_t = 1 + 0.9 \cdot 2 + 0.9^2 \cdot 10
]

[
G_t = 1 + 1.8 + 8.1 = 10.9
]

Same future rewards, different discount factor, different return.

---

## 6.5 Expected Return

The future is uncertain, so the agent usually cannot know exactly what return it will get.

Therefore, it maximizes expected return:

[
\mathbb{E}[G_t]
]

More precisely, under a policy (\pi):

[
\mathbb{E}\_\pi[G_t]
]

Meaning:

> Average return we expect if the agent follows policy (\pi).

This is the core RL objective:

[
\max*\pi \mathbb{E}*\pi[G_t]
]

In words:

> Find a policy that maximizes expected future reward.

---

## 6.6 Why Expectation Matters

Example:

Action A:

| Return | Probability |
| -----: | ----------: |
|     10 |         0.5 |
|      0 |         0.5 |

Expected return:

[
10 \cdot 0.5 + 0 \cdot 0.5 = 5
]

Action B:

| Return | Probability |
| -----: | ----------: |
|      4 |         1.0 |

Expected return:

[
4
]

Action A has higher expected return but is riskier.

This revisits Lecture 1’s expectation idea, now applied over time.

---

## 6.7 Delayed Reward

RL is hard because reward can be delayed.

Example:

In chess, most moves do not immediately give a reward.

The major reward comes at the end:

[
+1 = \text{win}
]

[
0 = \text{draw}
]

[
-1 = \text{loss}
]

The agent has to learn which earlier actions contributed to later success.

This is called the **credit assignment problem**.

Teaching sentence:

> RL is difficult because the consequence of an action may only become clear much later.

---

## 6.8 Mini-Exercise

An agent receives rewards:

[
R_{t+1} = 2,\quad R_{t+2}=4,\quad R_{t+3}=8
]

Compute return for:

[
\gamma = 0.5
]

Solution:

[
G_t = 2 + 0.5\cdot 4 + 0.5^2 \cdot 8
]

[
G_t = 2 + 2 + 2 = 6
]

Ask:

What happens if (\gamma) increases?

Expected answer:

Future rewards matter more, so the return becomes larger when future rewards are positive.

---

# Part 7 — Value Functions

**Time:** 8 minutes

## 7.1 Why Value Functions?

Expected return is central, but the agent needs a way to estimate which states and actions are good.

That is the role of value functions.

A value function estimates expected future reward.

---

## 7.2 State Value Function

The state value function is:

[
V^\pi(s) = \mathbb{E}\_\pi[G_t \mid S_t = s]
]

Meaning:

> If I am in state (s), and I follow policy (\pi), what return should I expect?

Interpretation:

[
V(s) = \text{how good is this state?}
]

Example:

In gridworld:

- states near the goal have high value,
- states near traps have low value,
- dead ends may have low value.

---

## 7.3 Action Value Function

The action value function is:

[
Q^\pi(s,a) = \mathbb{E}\_\pi[G_t \mid S_t = s, A_t = a]
]

Meaning:

> If I am in state (s), take action (a), and then follow policy (\pi), what return should I expect?

Interpretation:

[
Q(s,a) = \text{how good is this action in this state?}
]

This is extremely important because action selection can be based on (Q)-values.

Choose:

[
a = \arg\max_a Q(s,a)
]

---

## 7.4 Simple Q-Value Example

Suppose the robot is in state (s).

| Action | (Q(s,a)) |
| ------ | -------: |
| up     |      7.5 |
| right  |      4.0 |
| left   |     -1.0 |
| down   |      2.0 |

The greedy action is:

[
\text{up}
]

because it has the highest expected future return.

---

## 7.5 Value Is Not Just Immediate Reward

This is crucial.

An action can have low immediate reward but high value.

Example:

| Action             | Immediate Reward | Future Consequence | (Q(s,a)) |
| ------------------ | ---------------: | ------------------ | -------: |
| collect small coin |               +1 | dead end           |        1 |
| move toward goal   |                0 | later +10          |        8 |
| risky shortcut     |               -1 | maybe faster       |        5 |

The value function estimates long-term consequences.

Teaching point:

> (Q(s,a)) is not “reward now.” It is expected future return after taking that action.

---

## 7.6 Deep RL Connection

In tabular RL, you could store one value for each state or state-action pair.

But in real problems, there are too many states.

Example:

- images have millions of possible pixel configurations,
- robot sensors are continuous,
- conversations have enormous possible histories,
- games like Go have enormous state spaces.

So deep RL uses neural networks.

A neural network can approximate:

[
Q_\theta(s,a)
]

or a policy:

[
\pi_\theta(a \mid s)
]

Examples:

### DQN-style idea

Input:

[
s
]

Output:

[
Q(s,a_1), Q(s,a_2), ..., Q(s,a_k)
]

### Policy network

Input:

[
s
]

Output:

[
\pi(a_1 \mid s), \pi(a_2 \mid s), ..., \pi(a_k \mid s)
]

This connects RL back to deep learning.

---

## 7.7 Three Families of RL Methods

Briefly mention, not in detail.

### Value-based methods

Learn value functions.

Example:

[
Q(s,a)
]

Then choose actions with high value.

### Policy-based methods

Learn the policy directly.

Example:

[
\pi(a \mid s)
]

### Actor-critic methods

Learn both:

- actor = policy,
- critic = value estimator.

This is useful because many modern RL systems use actor-critic ideas.

Do not go into algorithmic detail unless time allows.

---

## 7.8 Mini-Exercise

Given:

| Action | Immediate reward | Estimated future reward after action | Total expected value |
| ------ | ---------------: | -----------------------------------: | -------------------: |
| A      |               +5 |                                    0 |                    5 |
| B      |                0 |                                   +8 |                    8 |
| C      |               -1 |                                  +10 |                    9 |

Questions:

1. Which action has the highest immediate reward?
2. Which action has the highest value?
3. Which action should a value-based agent choose?
4. What lesson does this teach?

Expected answers:

1. A.
2. C.
3. C.
4. Good decisions require considering future consequences, not just immediate rewards.

---

# Part 8 — Exploration vs Exploitation

**Time:** 3 minutes\*\*

## 8.1 The Tradeoff

The agent faces a central problem:

### Exploitation

Choose the action currently believed to be best.

Example:

[
a = \arg\max_a Q(s,a)
]

### Exploration

Try actions that might be worse now but could reveal useful information.

Example:

The robot tries a new path that might lead to the goal faster.

---

## 8.2 Why Exploration Is Necessary

If the agent only exploits what it currently believes, it may never discover better actions.

Example:

A recommender system keeps recommending the same type of video because it already knows the user likes it.

But it never learns whether the user might like another category even more.

In RL:

> Learning requires trying actions whose value is uncertain.

---

## 8.3 Probability in Exploration

Exploration often uses randomness.

### Epsilon-Greedy

With probability (1-\epsilon):

[
\text{choose best-known action}
]

With probability (\epsilon):

[
\text{choose random action}
]

Example:

[
\epsilon = 0.1
]

Means:

- 90% exploit,
- 10% explore.

### Softmax Action Selection

Actions with higher estimated value get higher probability, but lower-value actions may still be tried.

### Entropy Regularization

Encourages the policy to keep some randomness instead of becoming too deterministic too early.

Connection to Lecture 3:

> Entropy can measure how exploratory a policy is.

---

## 8.4 Mini-Exercise

An agent has Q-values:

| Action | (Q(s,a)) |
| ------ | -------: |
| left   |       10 |
| right  |        6 |
| wait   |        2 |

Questions:

1. Which action does pure exploitation choose?
2. Why might the agent still try “right” or “wait”?
3. What does (\epsilon = 0.1) mean in epsilon-greedy exploration?

Expected answers:

1. Left.
2. To learn whether its estimates are wrong or whether the environment changed.
3. 10% of the time, choose a random action.

---

# Part 9 — Lecture Summary

**Time:** 1 minute

## Core Ideas Students Should Remember

1. **RL is about sequential decisions under uncertainty.**

The agent acts, receives feedback, and changes future states.

[
S_t \rightarrow A_t \rightarrow R_{t+1}, S_{t+1}
]

---

2. **Transition probabilities describe how actions change the world.**

[
P(s' \mid s,a)
]

The same action can lead to different next states.

---

3. **The Markov property simplifies the future.**

[
P(S\_{t+1} \mid S_t,A_t,\text{history})
======================================

P(S\_{t+1} \mid S_t,A_t)
]

The current state should contain the relevant information from the past.

---

4. **A policy tells the agent how to act.**

Deterministic:

[
a = \pi(s)
]

Stochastic:

[
\pi(a \mid s)
]

A stochastic policy is a probability distribution over actions.

---

5. **The goal is expected future reward.**

[
G*t =
R*{t+1}

- \gamma R\_{t+2}
- \gamma^2 R\_{t+3}
- \cdots
  ]

[
\max*\pi \mathbb{E}*\pi[G_t]
]

---

6. **Value functions estimate expected return.**

[
V^\pi(s) = \mathbb{E}\_\pi[G_t \mid S_t=s]
]

[
Q^\pi(s,a) = \mathbb{E}\_\pi[G_t \mid S_t=s,A_t=a]
]

---

7. **Exploration requires controlled randomness.**

The agent must sometimes try uncertain actions to learn.

---

# Board Summary

A compact final board could look like this:

[
S_t \rightarrow A_t \rightarrow R_{t+1}, S_{t+1}
]

[
P(s' \mid s,a)
]

[
P(S\_{t+1} \mid S_t,A_t,\text{history})
======================================

P(S\_{t+1} \mid S_t,A_t)
]

[
\pi(a \mid s)
]

[
G*t =
R*{t+1}

- \gamma R\_{t+2}
- \gamma^2 R\_{t+3}
- \cdots
  ]

[
V^\pi(s) =
\mathbb{E}\_\pi[G_t \mid S_t=s]
]

[
Q^\pi(s,a) =
\mathbb{E}\_\pi[G_t \mid S_t=s,A_t=a]
]

[
\text{Goal: } \max*\pi \mathbb{E}*\pi[G_t]
]

And the AI translation:

| Probability/RL Concept | AI Meaning                         |
| ---------------------- | ---------------------------------- |
| State                  | current situation/context          |
| Action                 | model/agent choice                 |
| Reward                 | feedback signal                    |
| Transition probability | uncertain consequence of action    |
| Policy                 | distribution over actions          |
| Expected return        | long-term objective                |
| Value function         | estimate of future reward          |
| Exploration            | random action to learn             |
| Exploitation           | choose currently best-known action |

---

# Suggested Running Examples

## Running Example 1: Gridworld Robot

Use for:

- states,
- actions,
- rewards,
- transition probabilities,
- Markov property,
- expected return,
- value functions,
- exploration.

Example transition:

| Action | Next state | Probability |
| ------ | ---------- | ----------: |
| up     | cell above |         0.8 |
| up     | cell left  |         0.1 |
| up     | cell right |         0.1 |

---

## Running Example 2: Recommendation System

Use for:

- state as user context,
- action as recommended item,
- reward as click/watch/satisfaction,
- exploration vs exploitation.

Example:

| Action                   | Known reward estimate |
| ------------------------ | --------------------: |
| recommend sports video   |                  high |
| recommend cooking video  |               unknown |
| recommend politics video |                medium |

Teaching point:

> The system must balance showing what it already knows works with learning more about user preferences.

---

## Running Example 3: RLHF for LLMs

Use for:

- policy as language model,
- action as token/response,
- reward as preference score,
- expected reward as behavioral optimization.

Simplified formula:

[
\pi_\theta(\text{response} \mid \text{prompt})
]

Reward:

[
R(\text{prompt}, \text{response})
]

Objective:

[
\max\_\theta \mathbb{E}[R]
]

Teaching point:

> RLHF uses reward signals to shape the model’s output distribution beyond pure next-token imitation.

---

# Suggested Exercises

## Exercise 1 — Identify RL Components

A robot must deliver medicine in a hospital.

Questions:

1. What are possible states?
2. What are possible actions?
3. What could be the reward?
4. Why might transitions be stochastic?

Expected answers:

1. Location, battery, obstacles, delivery status, map.
2. Move, stop, turn, pick up, drop off.
3. Successful delivery, time penalty, collision penalty.
4. People move unpredictably, sensors are noisy, paths may be blocked.

---

## Exercise 2 — Transition Probabilities

A robot tries to move forward.

| Outcome       | Probability |
| ------------- | ----------: |
| moves forward |        0.75 |
| slips left    |        0.15 |
| slips right   |        0.10 |

Questions:

1. Write (P(s' \mid s,a)) for each outcome.
2. Do the probabilities sum to 1?
3. Is the environment deterministic or stochastic?

Expected answers:

1. (P(\text{forward} \mid s,\text{forward})=0.75), etc.
2. Yes.
3. Stochastic.

---

## Exercise 3 — Discounted Return

Rewards:

[
R_{t+1}=3,\quad R_{t+2}=2,\quad R_{t+3}=10
]

Discount:

[
\gamma = 0.5
]

Question:

Compute:

[
G_t
]

Solution:

[
G_t = 3 + 0.5 \cdot 2 + 0.5^2 \cdot 10
]

[
G_t = 3 + 1 + 2.5 = 6.5
]

---

## Exercise 4 — Policy Interpretation

A policy gives:

| Action | Probability |
| ------ | ----------: |
| up     |        0.50 |
| right  |        0.30 |
| left   |        0.10 |
| down   |        0.10 |

Questions:

1. Is the policy deterministic or stochastic?
2. What is (\pi(\text{right} \mid s))?
3. Which action is most likely?
4. Why might stochasticity help?

Expected answers:

1. Stochastic.
2. (0.30)
3. Up.
4. It allows exploration and avoids committing too early to one action.

---

## Exercise 5 — Value Function

Given:

| Action | (Q(s,a)) |
| ------ | -------: |
| up     |        2 |
| right  |        7 |
| left   |       -1 |
| down   |        4 |

Questions:

1. Which action has highest value?
2. What does (Q(s,\text{right})=7) mean?
3. Is (Q(s,a)) immediate reward or expected future return?

Expected answers:

1. Right.
2. If the agent takes right in state (s), it expects return 7 under the assumed policy/estimate.
3. Expected future return.

---

# What to Emphasize Most

The most important ideas in Lecture 4 are:

1. RL is not just prediction; it is action over time.
2. Actions influence future states and future data.
3. (P(s' \mid s,a)) is the probabilistic model of the environment.
4. (\pi(a \mid s)) is the probabilistic model of the agent’s behavior.
5. Return is cumulative future reward.
6. Expected return is the objective.
7. Value functions estimate expected future reward.
8. Exploration is necessary because the agent must learn from uncertain actions.

The deepest conceptual move is:

[
P(y \mid x)
]

from supervised learning becomes two major conditional structures in RL:

[
P(s' \mid s,a)
]

for the environment, and:

[
\pi(a \mid s)
]

for the agent.

---

# What Not to Overdo

Avoid spending too much time on:

- Bellman equation derivations,
- Q-learning update formulas,
- policy gradient derivations,
- PPO technical details,
- actor-critic algorithms,
- Monte Carlo vs temporal-difference learning in depth,
- continuous control math,
- off-policy vs on-policy distinctions,
- detailed RLHF implementation,
- causal inference issues in recommendation systems.

Those are important later, but for this crash course, the priority is conceptual and probabilistic fluency.

A good rule:

> Students should leave understanding what RL is optimizing, not necessarily how every RL algorithm optimizes it.

---

# Optional Advanced Add-On If Time Allows

If students are comfortable, add a short preview of the Bellman idea.

## Bellman Intuition

A state’s value equals:

1. immediate expected reward,
2. plus discounted value of the next state.

Informally:

[
\text{value now}
================

\text{reward now}

- \text{discounted value later}
  ]

More formally:

[
V(s)
====

\mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) \mid S_t=s]
]

For action values:

[
Q(s,a)
======

\mathbb{E}[R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') \mid S_t=s,A_t=a]
]

Do not derive it fully.

Just give the intuition:

> Value functions are recursive because the value of now depends on rewards now plus the value of what comes next.

This prepares students for deeper RL later.

---

# Recommended Ending

End by connecting Lecture 4 to Lecture 5:

> In this lecture, probability helped us understand decisions over time. The agent samples actions, the environment samples next states, and learning means maximizing expected future reward. In the next lecture, we move from decision-making to generation. We will see how sampling, latent variables, Gaussian noise, and denoising explain LLM decoding and diffusion models.

Final board line:

[
\text{RL: sample actions to maximize expected return}
]

[
\text{Generative AI: sample outputs from learned distributions}
]

Then say:

> Lecture 5 completes the course by showing how probability becomes generation: sampling tokens, sampling latent variables, and turning noise into images.
