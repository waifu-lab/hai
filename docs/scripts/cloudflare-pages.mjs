import { copyFile, writeFile } from "node:fs/promises";

const out = new URL("../.output/public/", import.meta.url);

await copyFile(new URL("_shell.html", out), new URL("index.html", out));

await writeFile(
  new URL("_headers", out),
  ["/api/search", "  Content-Type: application/json; charset=utf-8", ""].join(
    "\n",
  ),
);
