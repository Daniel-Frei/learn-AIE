export const ALL_TOPICS = ["RL", "DL", "NLP", "Math", "Life Science"] as const;

export type Topic = (typeof ALL_TOPICS)[number];
