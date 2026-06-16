import { expect, test, type Page } from "@playwright/test";

function trackSelectionPermissionErrors(page: Page) {
  const errors: string[] = [];
  const selectionPermissionPattern =
    /Permission denied to access property "(?:__reactFiber|correspondingUseElement)/i;

  page.on("console", (message) => {
    if (
      message.type() === "error" &&
      selectionPermissionPattern.test(message.text())
    ) {
      errors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    if (selectionPermissionPattern.test(error.message)) {
      errors.push(error.message);
    }
  });

  return errors;
}

test("lists available learning courses", async ({ page }) => {
  await page.goto("/learn");

  await expect(
    page.getByRole("heading", { name: /choose a course/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /stanford cme295 transformers & llms/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /crash course probability/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /clinical trials crash course/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /crash course medicine/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /text into transformers/i }),
  ).toHaveCount(0);
});

test("lists learning experiences for a selected course", async ({ page }) => {
  await page.goto("/learn/crash-course-probability");

  await expect(
    page.getByRole("heading", { name: /crash course probability/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /likelihood, loss, softmax/i }),
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /rl over time/i })).toBeVisible();
  await expect(
    page.getByRole("link", { name: /generation sampling lab/i }),
  ).toBeVisible();

  await page.getByRole("link", { name: /rl over time/i }).click();
  await expect(page).toHaveURL(
    /\/learn\/crash-course-probability\/crash-probability-l4$/,
  );
});

test("lists the standalone Medicine learning page for its course", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-medicine");

  await expect(
    page.getByRole("heading", { name: /crash course medicine/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", {
      name: /what medicine is/i,
    }),
  ).toBeVisible();

  await page.getByRole("link", { name: /what medicine is/i }).click();
  await expect(page).toHaveURL(/\/learn\/crash-course-medicine\/lecture-1$/);
});

test("labels and sorts course learning experiences by source sequence", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  const courseCards = page.locator(
    'section[aria-label="Stanford CME295 Transformers & LLMs learning experiences"] a',
  );
  await expect(courseCards).toHaveCount(6);

  const cardTexts = await courseCards.allTextContents();
  expect(cardTexts[0]).toContain("Lecture 1");
  expect(cardTexts[1]).toContain("Lecture 2");
  expect(cardTexts[2]).toContain("Lecture 3");
  expect(cardTexts[3]).toContain("Lecture 4");
  expect(cardTexts[4]).toContain("Lecture 5");
  expect(cardTexts[5]).toContain("Lecture 6");
  expect(cardTexts[5]).toContain("Reasoning Control Bench");
  expect(cardTexts.join("\n")).not.toMatch(/15 min|16 min|18 min|19 min/i);
  expect(cardTexts.join("\n")).not.toMatch(
    /Introductory NLP with ML basics|After CME295 Lecture/i,
  );
});

test("opens the standalone Stanford CME295 Lecture 6 page from the course page", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295");

  await page.getByRole("link", { name: /reasoning control bench/i }).click();
  await expect(page).toHaveURL(/\/learn\/stanford-cme295\/lecture-6$/);
  await expect(
    page.getByRole("heading", {
      name: /run a reasoning model like a controlled experiment/i,
    }),
  ).toBeVisible();
});

test("keeps legacy direct learning source URLs working", async ({ page }) => {
  await page.goto("/learn/crash-probability-l3");

  await expect(
    page.getByRole("heading", {
      name: /make the observed answer more probable/i,
    }),
  ).toBeVisible();
});

test("renders the Stanford CME295 Lecture 1 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await expect(
    page.getByRole("heading", { name: /follow text into a transformer/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /tokenization tradeoff/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*QK/)).toHaveCount(0);

  const tokenizationExplorer = page.getByTestId("tokenization-explorer");
  await tokenizationExplorer.getByRole("button", { name: /^Word$/i }).click();
  await expect(tokenizationExplorer.getByRole("status")).toHaveText(
    /unknown token/i,
  );

  const attentionExplorer = page.getByTestId("attention-focus-explorer");
  await attentionExplorer
    .getByRole("button", { name: /query: reading/i })
    .click();
  await expect(attentionExplorer.getByText(/top focus: bear/i)).toBeVisible();

  const qkvCheck = page.getByTestId("qkv-check");
  await qkvCheck.getByRole("button", { name: /values are compared/i }).click();
  await expect(qkvCheck.getByRole("status")).toHaveText(/not yet/i);

  await qkvCheck.getByRole("button", { name: /queries are compared/i }).click();
  await expect(qkvCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 2 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect2");

  await expect(
    page.getByRole("heading", { name: /tune the transformer upgrade knobs/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /position and attention design bench/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /bert input and fine-tuning lab/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*PE/)).toHaveCount(0);

  const positionBench = page.getByTestId("position-attention-bench");
  await positionBench.getByRole("button", { name: /^ALiBi$/i }).click();
  await expect(positionBench.getByRole("status")).toHaveText(
    /deterministic linear distance penalty/i,
  );

  const efficiencyLab = page.getByTestId("attention-efficiency-lab");
  await efficiencyLab.getByRole("button", { name: /^MQA$/i }).click();
  await expect(efficiencyLab.getByRole("status")).toHaveText(
    /all heads share keys and values/i,
  );

  const bertLab = page.getByTestId("bert-input-lab");
  await bertLab.getByRole("button", { name: /token task/i }).click();
  await bertLab.getByRole("button", { name: /^Output$/i }).click();
  await expect(bertLab.getByRole("status")).toHaveText(
    /each non-special token output/i,
  );

  const positionCheck = page.getByTestId("position-extrapolation-check");
  await positionCheck
    .getByRole("button", { name: /softmax cannot normalize/i })
    .click();
  await expect(positionCheck.getByRole("status")).toHaveText(/not yet/i);

  await positionCheck
    .getByRole("button", { name: /no learned position vectors/i })
    .click();
  await expect(positionCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 3 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect3");

  await expect(
    page.getByRole("heading", {
      name: /follow one request through llm inference/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /trace one request through the llm runtime/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /place each technique before comparing tradeoffs/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*P\(t_i\)/)).toHaveCount(0);

  const runtimeLab = page.getByTestId("llm-runtime-lab");
  await runtimeLab.getByRole("button", { name: /next-token policy/i }).click();
  await runtimeLab.getByRole("button", { name: /Guided JSON/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(/external grammar/i);

  await runtimeLab.getByRole("button", { name: /MoE/i }).click();
  await runtimeLab.getByRole("button", { name: /^Collapsed$/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(
    /Most tokens route to one expert/i,
  );

  await runtimeLab.getByRole("button", { name: /Cache/i }).click();
  await runtimeLab.getByRole("button", { name: /^PagedAttention$/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(
    /non-contiguous pages/i,
  );

  await runtimeLab.getByRole("button", { name: /Accelerate/i }).click();
  await runtimeLab.getByRole("button", { name: /^MTP$/i }).click();
  await expect(runtimeLab.getByRole("status")).toHaveText(
    /extra heads predict future positions/i,
  );

  const routingCheck = page.getByTestId("routing-collapse-check");
  await routingCheck
    .getByRole("button", { name: /top-p sampling instead of beam search/i })
    .click();
  await expect(routingCheck.getByRole("status")).toHaveText(/not yet/i);

  await routingCheck
    .getByRole("button", { name: /router has collapsed/i })
    .click();
  await expect(routingCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 4 learning page and supports interactions", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect4");

  await expect(
    page.getByRole("heading", { name: /run the llm training pipeline/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /follow one model from raw continuation to tuned behavior/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /match the memory object to the optimization lever/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*C_\{\\mathrm/)).toHaveCount(0);

  const lifecycle = page.getByTestId("training-lifecycle-workbench");
  await lifecycle.getByRole("button", { name: /stage 3\s+sft/i }).click();
  await expect(lifecycle.getByRole("status")).toHaveText(
    /instruction-tuned model/i,
  );

  const scalingLab = page.getByTestId("scaling-budget-lab");
  await scalingLab
    .getByRole("button", { name: /gpt-3-style example/i })
    .click();
  await expect(scalingLab.getByRole("status")).toHaveText(/undertrained/i);

  const memoryLab = page.getByTestId("memory-optimization-lab");
  await memoryLab.getByRole("button", { name: /^FlashAttention$/i }).click();
  await expect(memoryLab.getByRole("status")).toHaveText(/exact attention/i);

  const loraWorkbench = page.getByTestId("lora-workbench");
  await loraWorkbench.getByRole("button", { name: /^QLoRA$/i }).click();
  await expect(loraWorkbench.getByRole("status")).toHaveText(
    /frozen W0 is stored quantized/i,
  );

  const flashCheck = page.getByTestId("flashattention-check");
  await flashCheck
    .getByRole("button", { name: /approximates attention/i })
    .click();
  await expect(flashCheck.getByRole("status")).toHaveText(/not yet/i);
  await flashCheck
    .getByRole("button", { name: /reduces slow hbm reads/i })
    .click();
  await expect(flashCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Stanford CME295 Lecture 5 learning page and supports preference-tuning labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect5");

  await expect(
    page.getByRole("heading", {
      name: /tune preferences without chasing the proxy/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /turn a vague complaint into a training example/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /reward modeling turns pairs into score differences/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /read the alignment pipeline before the algorithms/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /map language generation into rl before using rlhf/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /ppo means proximal policy optimization/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /dpo compares what changed relative to the reference/i,
    }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*P\(y_w/)).toHaveCount(0);
  await expect(page.getByText(/State s_t/)).toHaveCount(0);
  await expect(page.getByText(/Action a_t/)).toHaveCount(0);
  await expect(page.getByText(/pi\(a_t \| s_t\)/)).toHaveCount(0);
  await expect(page.getByText(/Advantage A_t/)).toHaveCount(0);

  const preferenceStudio = page.getByTestId("preference-data-studio");
  await preferenceStudio.getByRole("button", { name: /^Pointwise/i }).click();
  await expect(preferenceStudio.getByRole("status")).toHaveText(
    /absolute 0\.9/i,
  );
  await preferenceStudio.getByRole("button", { name: /^Pairwise/i }).click();
  await expect(preferenceStudio.getByRole("status")).toHaveText(
    /choosing the better of two outputs/i,
  );

  const rewardLab = page.getByTestId("bradley-terry-lab");
  await rewardLab.getByRole("button", { name: /equal rewards/i }).click();
  await expect(rewardLab.getByRole("status")).toHaveText(
    /0\.50 preference probability/i,
  );

  const ppoLab = page.getByTestId("ppo-clip-lab");
  await expect(ppoLab.getByText(/0\.30 \/ 0\.20 = 1\.50/)).toBeVisible();
  await expect(ppoLab.getByText(/min\(3\.00, 2\.40\) = 2\.40/)).toBeVisible();
  await ppoLab.getByRole("button", { name: /negative advantage/i }).click();
  await expect(ppoLab.getByRole("status")).toHaveText(
    /limits how far probability can drop/i,
  );

  const bestOfNLab = page.getByTestId("best-of-n-lab");
  await expect(
    bestOfNLab.getByText(/1 - 0\.70\^4 = 0\.76/).first(),
  ).toBeVisible();
  await bestOfNLab.getByRole("button", { name: /N=8/i }).click();
  await expect(bestOfNLab.getByRole("status")).toHaveText(/roughly 8x/i);
  await expect(
    bestOfNLab.getByText(/1 - 0\.70\^8 = 0\.94/).first(),
  ).toBeVisible();

  const dpoLab = page.getByTestId("dpo-logit-lab");
  await expect(dpoLab.getByText(/0\.10 \* 1\.50 = 0\.15/)).toBeVisible();
  await dpoLab.getByRole("button", { name: /policy favors rejected/i }).click();
  await expect(dpoLab.getByRole("status")).toHaveText(/goes below zero/i);

  const rewardHackingCheck = page.getByTestId("reward-hacking-check");
  await rewardHackingCheck
    .getByRole("button", { name: /learned new facts/i })
    .click();
  await expect(rewardHackingCheck.getByRole("status")).toHaveText(/not yet/i);
  await rewardHackingCheck
    .getByRole("button", { name: /exploited a proxy reward/i })
    .click();
  await expect(rewardHackingCheck.getByRole("status")).toHaveText(/correct/i);

  const dpoCheck = page.getByTestId("dpo-check");
  await dpoCheck
    .getByRole("button", { name: /no longer needs preference data/i })
    .click();
  await expect(dpoCheck.getByRole("status")).toHaveText(/not yet/i);
  await dpoCheck
    .getByRole("button", { name: /preference pairs and a frozen reference/i })
    .click();
  await expect(dpoCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the standalone Stanford CME295 Lecture 6 page and supports reasoning labs", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-6");

  await expect(
    page.getByRole("heading", {
      name: /run a reasoning model like a controlled experiment/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /reasoning means solving through intermediate steps/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /pass@k answers a different question than pass@1/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /grpo grades a response against its sampled group/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /ppo and grpo share update control but differ in the baseline/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /start reasoning questions/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[.*A_i/)).toHaveCount(0);

  const primer = page.getByTestId("reasoning-primer");
  await primer.getByRole("button", { name: /direct answer/i }).click();
  await expect(primer.getByRole("status")).toHaveText(/little evidence/i);
  await primer
    .getByRole("button", { name: /hidden trace with summary/i })
    .click();
  await expect(primer.getByRole("status")).toHaveText(/summary/i);

  const capabilityLab = page.getByTestId("capability-bottleneck-lab");
  await page.getByRole("button", { name: /course-code lookup/i }).click();
  await capabilityLab.getByRole("button", { name: /^Reasoning$/i }).click();
  await expect(capabilityLab.getByRole("status")).toHaveText(/try again/i);
  await capabilityLab
    .getByRole("button", { name: /knowledge lookup/i })
    .click();
  await expect(capabilityLab.getByRole("status")).toHaveText(
    /correct bottleneck/i,
  );

  const benchmarkLab = page.getByTestId("benchmark-lab");
  await benchmarkLab.getByRole("button", { name: /k=8/i }).click();
  await expect(benchmarkLab.getByRole("status")).toHaveText(
    /1 - C\(6, 8\) \/ C\(10, 8\) = 1 - 0 \/ 45 = 1\.00/i,
  );
  await expect(
    benchmarkLab.getByText(/cons@k samples several answers/i),
  ).toBeVisible();
  await benchmarkLab.getByRole("button", { name: /split vote/i }).click();
  await expect(benchmarkLab.getByText(/weak plurality/i)).toBeVisible();

  const budgetLab = page.getByTestId("reasoning-budget-lab");
  await budgetLab.getByRole("button", { name: /simple lookup/i }).click();
  await expect(budgetLab.getByRole("status")).toHaveText(
    /extra thinking mostly adds latency/i,
  );

  const rewardLab = page.getByTestId("verifiable-reward-lab");
  await rewardLab.getByRole("button", { name: /open-ended advice/i }).click();
  await expect(rewardLab.getByRole("status")).toHaveText(
    /not directly verifiable/i,
  );

  const grpoLab = page.getByTestId("grpo-group-lab");
  await expect(grpoLab.getByText(/GRPO objective skeleton/i)).toBeVisible();
  await grpoLab.getByRole("button", { name: /wrong but long/i }).click();
  await expect(grpoLab.getByRole("status")).toHaveText(
    /should not be rewarded for length/i,
  );

  const comparisonLab = page.getByTestId("ppo-grpo-comparison-lab");
  await comparisonLab.getByRole("button", { name: /^PPO$/i }).click();
  await expect(comparisonLab.getByRole("status")).toHaveText(
    /value model can reduce variance/i,
  );
  await comparisonLab.getByRole("button", { name: /^GRPO$/i }).click();
  await expect(comparisonLab.getByRole("status")).toHaveText(
    /avoiding the value model simplifies/i,
  );

  const lengthLab = page.getByTestId("length-incentive-lab");
  await expect(lengthLab.getByRole("status")).toHaveText(/5\.0x/i);
  await lengthLab
    .getByRole("button", { name: /equalized contribution/i })
    .click();
  await expect(lengthLab.getByRole("status")).toHaveText(/reasoning quality/i);

  const r1Pipeline = page.getByTestId("r1-pipeline-lab");
  await r1Pipeline.getByRole("button", { name: /^R1$/i }).click();
  await expect(r1Pipeline.getByRole("status")).toHaveText(
    /small reasoning sft cold start/i,
  );

  const distillationLab = page.getByTestId("distillation-lab");
  await distillationLab
    .getByRole("button", { name: /ordinary distillation/i })
    .click();
  await expect(distillationLab.getByRole("status")).toHaveText(
    /distributional knowledge/i,
  );

  const misconceptionCheck = page.getByTestId("reasoning-misconception-check");
  await misconceptionCheck
    .getByRole("button", { name: /always force extended thinking/i })
    .click();
  await expect(misconceptionCheck.getByRole("status")).toHaveText(/not quite/i);
  await misconceptionCheck
    .getByRole("button", { name: /spend more thinking/i })
    .click();
  await expect(misconceptionCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Probability L3 learning page and supports checks", async ({
  page,
}) => {
  const selectionPermissionErrors = trackSelectionPermissionErrors(page);

  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await expect(
    page.getByRole("heading", {
      name: /make the observed answer more probable/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /softmax explorer/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\$\$P\(y_i\)/)).toHaveCount(0);

  const logitCheck = page.getByTestId("logit-check");
  await logitCheck
    .getByRole("button", { name: /already probabilities/i })
    .click();
  await expect(logitCheck.getByRole("status")).toHaveText(/not yet/i);

  await logitCheck.getByRole("button", { name: /highest raw score/i }).click();
  await expect(logitCheck.getByRole("status")).toHaveText(/correct/i);

  const heroSummary = page.getByText(/neural-network training looks less/i);
  const box = await heroSummary.boundingBox();
  expect(box, "learning text should be visible").not.toBeNull();
  if (box) {
    await page.mouse.move(box.x + 2, box.y + box.height / 2);
    await page.mouse.down();
    await page.mouse.move(
      box.x + Math.min(box.width - 2, 360),
      box.y + box.height / 2,
      { steps: 12 },
    );
    await page.mouse.up();
  }

  expect(
    selectionPermissionErrors,
    "selecting learning text should not produce React permission errors",
  ).toEqual([]);
});

test("renders the Probability L4 learning page and supports gridworld checks", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l4");

  await expect(
    page.getByRole("heading", {
      name: /choose actions by averaging possible futures/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /gridworld expected-return lab/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /epsilon-greedy policy lab/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\$\$P\(s'/)).toHaveCount(0);

  const gridworldLab = page.getByTestId("gridworld-decision-lab");
  await gridworldLab.getByRole("button", { name: /^Up$/i }).click();
  await expect(page.getByTestId("gridworld-summary")).toHaveText(
    /selected action: up/i,
  );

  const markovCheck = page.getByTestId("markov-check");
  await markovCheck
    .getByRole("button", { name: /latest message is the current observation/i })
    .click();
  await expect(markovCheck.getByRole("status")).toHaveText(/not yet/i);

  await markovCheck
    .getByRole("button", { name: /includes relevant history or memory/i })
    .click();
  await expect(markovCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the Probability L5 learning page and supports generation labs", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l5");

  await expect(
    page.getByRole("heading", {
      name: /turn uncertainty into generated output/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /sampling distribution lab/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /generation is a distribution, a rule, then a draw/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /what each control means/i }),
  ).toBeVisible();
  await expect(page.getByText(/what the lab numbers mean/i)).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /latent house probe/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /marginalize over hidden alternatives/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /forward noise first, learned reverse process second/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /diffusion denoising path/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\$\$x\\sim/)).toHaveCount(0);

  const samplingLab = page.getByTestId("sampling-lab");
  await samplingLab.getByRole("button", { name: /^Greedy$/i }).click();
  await expect(page.getByTestId("sampling-summary")).toHaveText(
    /greedy selects mat/i,
  );
  await samplingLab.getByRole("button", { name: /draw next sample/i }).click();
  await samplingLab.getByRole("button", { name: /^Top-k$/i }).click();
  await expect(page.getByTestId("sampling-summary")).toHaveText(
    /top-k selects mat/i,
  );

  const temperatureCheck = page.getByTestId("temperature-knowledge-check");
  await temperatureCheck
    .getByRole("button", { name: /adds new knowledge/i })
    .click();
  await expect(temperatureCheck.getByRole("status")).toHaveText(/not yet/i);
  await temperatureCheck
    .getByRole("button", { name: /flattens the distribution/i })
    .click();
  await expect(temperatureCheck.getByRole("status")).toHaveText(/correct/i);

  const latentProbe = page.getByTestId("latent-variable-probe");
  await latentProbe
    .getByLabel(/probe style latent factor/i)
    .selectOption("modern");
  await expect(latentProbe.getByText(/style: Modern/i)).toBeVisible();
  await expect(latentProbe.getByText(/flat roof/i)).toBeVisible();

  const diffusionLab = page.getByTestId("diffusion-path-lab");
  await diffusionLab.getByRole("button", { name: /Seed B/i }).click();
  await diffusionLab.getByRole("button", { name: /Denoise one step/i }).click();
  await expect(page.getByTestId("diffusion-summary")).toHaveText(
    /Seed B \/ t=4/i,
  );
});

test("keeps the learning page usable at mobile width", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 1 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 2 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect2");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 3 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect3");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 4 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect4");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Stanford CME295 Lecture 5 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/cme295-lect5");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the standalone Stanford CME295 Lecture 6 page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/stanford-cme295/lecture-6");

  await expect(
    page.getByRole("link", { name: /back to stanford cme295 course/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Probability L4 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-probability/crash-probability-l4");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the Probability L5 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-probability/crash-probability-l5");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("renders the Clinical Trials L3 learning page and supports checks", async ({
  page,
}) => {
  await page.goto("/learn/clinical-trials/clinical-trials-l3");

  await expect(
    page.getByRole("heading", {
      name: /read the size, uncertainty, and clinical meaning/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: /treatment effect explorer/i }),
  ).toBeVisible();
  await expect(page.locator(".katex-display").first()).toBeVisible();
  await expect(page.getByText(/\\\[\\text\{ARR\}/)).toHaveCount(0);

  const absoluteRiskCheck = page.getByTestId("absolute-risk-check");
  await absoluteRiskCheck
    .getByRole("button", { name: /always clinically large/i })
    .click();
  await expect(absoluteRiskCheck.getByRole("status")).toHaveText(/not yet/i);

  await absoluteRiskCheck
    .getByRole("button", { name: /absolute benefit/i })
    .click();
  await expect(absoluteRiskCheck.getByRole("status")).toHaveText(/correct/i);
});

test("renders the standalone Medicine Lecture 1 page and supports clinical reasoning interactions", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-medicine/lecture-1");

  await expect(
    page.getByRole("heading", {
      name: /what medicine is/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", {
      name: /work through a patient case in clinical order/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toHaveCount(0);

  const studio = page.getByTestId("medicine-clinical-studio");
  await studio.getByRole("button", { name: /exam and vitals/i }).click();
  await expect(
    studio.getByRole("button", { name: /^Oxygen saturation 89%$/i }),
  ).toBeVisible();

  await studio.getByRole("button", { name: /^model$/i }).click();
  await expect(studio.locator(".representation-output")).toContainText(
    /hypoxemia/i,
  );

  await studio.getByRole("button", { name: /give oxygen/i }).click();
  await expect(
    studio.getByText(/low oxygen is a direct instability signal/i),
  ).toBeVisible();

  await studio.getByRole("button", { name: /wait for every test/i }).click();
  await expect(studio.getByText(/premature closure is active/i)).toBeVisible();
});

test("keeps the Clinical Trials L3 learning page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/clinical-trials/clinical-trials-l3");

  await expect(
    page.getByRole("link", { name: /start questions/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("keeps the standalone Medicine Lecture 1 page usable at mobile width", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/learn/crash-course-medicine/lecture-1");

  await expect(
    page.getByRole("link", { name: /back to medicine course/i }),
  ).toBeVisible();
  const scrollWidth = await page.evaluate(
    () => document.documentElement.scrollWidth,
  );
  const viewportWidth = await page.evaluate(() => window.innerWidth);
  expect(scrollWidth).toBeLessThanOrEqual(viewportWidth + 1);
});

test("transitions from Stanford CME295 Lecture 1 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect1");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect1$/);
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 1: transformers & llms/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 69/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 2 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect2");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect2$/);
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 2: transformer-based models & tricks/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 85/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 3 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect3$/);
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 3: large language models, moe & inference/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 80/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 4 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect4");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect4$/);
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 4: llm training, scaling & alignment/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 80/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 Lecture 5 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/cme295-lect5");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect5$/);
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 5: llm preference tuning, rlhf & dpo/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Stanford CME295 reasoning learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme295/lecture-6");

  await page.getByRole("link", { name: /start reasoning questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=cme295-lect6$/);
  await expect(
    page.getByRole("heading", {
      name: /stanford cme295 lecture 6: llm reasoning & test-time scaling/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from learning into the matching quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l3$/);
  await expect(
    page.getByRole("heading", {
      name: /crash course probability l3: likelihood, loss, softmax, and deep learning/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Probability L4 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l4");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l4$/);
  await expect(
    page.getByRole("heading", {
      name: /crash course probability l4: probability over time: reinforcement learning/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Probability L5 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/crash-course-probability/crash-probability-l5");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=crash-probability-l5$/);
  await expect(
    page.getByRole("heading", {
      name: /crash course probability l5: sampling, latent variables, and diffusion models/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});

test("transitions from Clinical Trials L3 learning into its quiz source", async ({
  page,
}) => {
  await page.goto("/learn/clinical-trials/clinical-trials-l3");

  await page.getByRole("link", { name: /start questions/i }).click();

  await expect(page).toHaveURL(/\/\?source=clinical-trials-l3$/);
  await expect(
    page.getByRole("heading", {
      name: /clinical trials crash course l3: statistics and evidence interpretation/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /choose filters/i }),
  ).toBeVisible();
  await expect(page.getByText(/question 1 of 60/i)).toBeVisible({
    timeout: 10000,
  });
});
