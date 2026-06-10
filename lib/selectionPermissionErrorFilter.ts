const selectionPermissionMessageSignals = [
  'Permission denied to access property "__reactFiber',
  'Permission denied to access property "correspondingUseElement"',
] as const;

const reactEventStackSignals = [
  "getClosestInstanceFromNode",
  "getEventTarget",
  "findInstanceBlockingEvent",
  "dispatchContinuousEvent",
  "dispatchEventForPluginEventSystem",
] as const;

export function isReactSelectionPermissionError(
  message: string,
  stack = "",
): boolean {
  const hasKnownMessage = selectionPermissionMessageSignals.some((signal) =>
    message.includes(signal),
  );

  if (!hasKnownMessage) return false;
  if (!stack) return true;

  return reactEventStackSignals.some((signal) => stack.includes(signal));
}

export const reactSelectionPermissionErrorFilterScript = `
(() => {
  const messageSignals = ${JSON.stringify(selectionPermissionMessageSignals)};
  const stackSignals = ${JSON.stringify(reactEventStackSignals)};

  const hasAnySignal = (value, signals) =>
    signals.some((signal) => String(value || "").includes(signal));

  const getErrorLike = (event) => event && (event.error || event.reason);

  const getMessage = (event) => {
    const errorLike = getErrorLike(event);
    if (event && event.message) return String(event.message);
    if (errorLike && errorLike.message) return String(errorLike.message);
    return String(errorLike || "");
  };

  const getStack = (event) => {
    const errorLike = getErrorLike(event);
    return errorLike && errorLike.stack ? String(errorLike.stack) : "";
  };

  const isReactSelectionPermissionError = (event) => {
    const message = getMessage(event);
    const stack = getStack(event);

    if (!hasAnySignal(message, messageSignals)) return false;
    return !stack || hasAnySignal(stack, stackSignals);
  };

  const suppressKnownReactSelectionPermissionError = (event) => {
    if (!isReactSelectionPermissionError(event)) return;

    event.preventDefault();
    event.stopImmediatePropagation();
  };

  window.addEventListener(
    "error",
    suppressKnownReactSelectionPermissionError,
    true,
  );
  window.addEventListener(
    "unhandledrejection",
    suppressKnownReactSelectionPermissionError,
    true,
  );

  Object.defineProperty(window, "__learningAiReactSelectionErrorFilter", {
    configurable: true,
    value: true,
  });
})();
`;
