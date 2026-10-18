import { DynamicCodeBlock } from "fumadocs-ui/components/dynamic-codeblock.core";
import { createHighlighterCore, type HighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

// 預設的 DynamicCodeBlock 會載入 Oniguruma 的 wasm，Vite 打包不了，改用 JS 正規表示式引擎。
let highlighter: Promise<HighlighterCore> | undefined;

function getHighlighter() {
  highlighter ??= createHighlighterCore({
    engine: createJavaScriptRegexEngine(),
    langs: [import("shiki/langs/python.mjs")],
    themes: [
      import("shiki/themes/github-light.mjs"),
      import("shiki/themes/github-dark.mjs"),
    ],
  });
  return highlighter;
}

export function PythonCode({ code }: { code: string }) {
  return (
    <DynamicCodeBlock
      lang="python"
      code={code}
      highlighter={getHighlighter}
      options={{ themes: { light: "github-light", dark: "github-dark" } }}
    />
  );
}
