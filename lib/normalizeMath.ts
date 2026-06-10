export function normalizeMathDelimiters(text: string): string {
  const normalizedEscapes = text.replace(/\\\\(?=[A-Za-z()[\]])/g, "\\");

  return normalizedEscapes
    .replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, expression: string) => {
      return `\n\n$$\n${expression.trim()}\n$$\n\n`;
    })
    .replace(/\$\$([^\n$][^\n]*?)\$\$/g, (_, expression: string) => {
      return `\n\n$$\n${expression.trim()}\n$$\n\n`;
    })
    .replace(/\\\(/g, () => "$")
    .replace(/\\\)/g, () => "$");
}
