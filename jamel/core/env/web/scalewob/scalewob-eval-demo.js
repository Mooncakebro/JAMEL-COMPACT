#!/usr/bin/env node

const fs = require("fs");
const http = require("http");
const net = require("net");
const path = require("path");
const crypto = require("crypto");
const { URL } = require("url");
const { spawn } = require("child_process");

const ROOT = path.resolve(
  process.env.SCALEWOB_ROOT ||
    path.join(__dirname, "../../../../../env/browser_env/scalewob-env"),
);

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (!token.startsWith("--")) {
      continue;
    }
    const key = token.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
      continue;
    }
    args[key] = next;
    i += 1;
  }
  return args;
}

function usage() {
  return `
Usage:
  node jamel/core/env/web/scalewob/scalewob-eval-demo.js --env <env-id> [options]

Options:
  --env <id>              Environment directory, for example: weibo, slack, xiaoheihe
  --task-id <n>           Task id to pass into the evaluator
  --params <json>         Extra evaluation params as a JSON object
  --params-file <file>    File containing evaluation params JSON
  --before <js>           JS to run in the page before evaluation
  --before-file <file>    File containing JS to run before evaluation
  --port <n>              Local static server port, default: 8787
  --cdp-port <n>          Browser remote debugging port, default: 9222
  --browser <path>        Optional browser binary to launch
  --user-data-dir <dir>   Optional browser profile dir, default: /tmp/scalewob-cdp-<env>
  --timeout-ms <n>        Timeout for page/CDP polling, default: 20000
  --close-browser         Close the launched browser when finished
  --help                  Show this help

Examples:
  node jamel/core/env/web/scalewob/scalewob-eval-demo.js --env weibo

  node jamel/core/env/web/scalewob/scalewob-eval-demo.js \\
    --env xiaoheihe \\
    --task-id 5 \\
    --params '{"bio":"Hardcore Gamer"}' \\
    --before-file jamel/core/env/web/scalewob/xiaoheihe-task5-before.js
`.trim();
}

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function parseJson(label, value) {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`Failed to parse ${label} as JSON: ${error.message}`);
  }
}

function loadParams(args) {
  let params = {};
  if (args.params) {
    params = parseJson("--params", args.params);
  } else if (args["params-file"]) {
    params = parseJson(args["params-file"], readText(path.resolve(ROOT, args["params-file"])));
  }
  if (params === null || Array.isArray(params) || typeof params !== "object") {
    throw new Error("Evaluation params must be a JSON object");
  }
  if (args["task-id"] !== undefined) {
    params.taskId = Number(args["task-id"]);
  }
  return params;
}

function loadBeforeScript(args) {
  if (args.before) {
    return args.before;
  }
  if (args["before-file"]) {
    return readText(path.resolve(ROOT, args["before-file"]));
  }
  return "";
}

function mimeType(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js")) return "application/javascript; charset=utf-8";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".json")) return "application/json; charset=utf-8";
  if (filePath.endsWith(".svg")) return "image/svg+xml";
  if (filePath.endsWith(".png")) return "image/png";
  if (filePath.endsWith(".jpg") || filePath.endsWith(".jpeg")) return "image/jpeg";
  return "application/octet-stream";
}

function createStaticServer(rootDir, port) {
  const server = http.createServer((req, res) => {
    const reqUrl = new URL(req.url, `http://${req.headers.host}`);
    let filePath = path.join(rootDir, decodeURIComponent(reqUrl.pathname));
    if (reqUrl.pathname === "/") {
      filePath = path.join(rootDir, "index.html");
    }

    if (!filePath.startsWith(rootDir)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }

    fs.stat(filePath, (statError, stat) => {
      if (statError) {
        res.writeHead(404);
        res.end("Not Found");
        return;
      }

      const finalPath = stat.isDirectory()
        ? path.join(filePath, "index.html")
        : filePath;

      fs.readFile(finalPath, (readError, data) => {
        if (readError) {
          res.writeHead(404);
          res.end("Not Found");
          return;
        }

        res.writeHead(200, { "Content-Type": mimeType(finalPath) });
        res.end(data);
      });
    });
  });

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => {
      resolve(server);
    });
  });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function poll(fn, timeoutMs, label) {
  const started = Date.now();
  let lastError;
  while (Date.now() - started < timeoutMs) {
    try {
      const value = await fn();
      if (value) {
        return value;
      }
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  const suffix = lastError ? ` Last error: ${lastError.message}` : "";
  throw new Error(`Timed out waiting for ${label}.${suffix}`);
}

function httpGetJson(urlString) {
  return new Promise((resolve, reject) => {
    const req = http.get(urlString, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () => {
        if (res.statusCode !== 200) {
          reject(new Error(`HTTP ${res.statusCode} from ${urlString}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.on("error", reject);
  });
}

function httpRequestJson(urlString, method) {
  return new Promise((resolve, reject) => {
    const target = new URL(urlString);
    const req = http.request(
      {
        method,
        hostname: target.hostname,
        port: target.port,
        path: `${target.pathname}${target.search}`,
      },
      (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          body += chunk;
        });
        res.on("end", () => {
          if (res.statusCode !== 200) {
            reject(new Error(`HTTP ${res.statusCode} from ${urlString}`));
            return;
          }
          try {
            resolve(JSON.parse(body));
          } catch (error) {
            reject(error);
          }
        });
      },
    );
    req.on("error", reject);
    req.end();
  });
}

function launchBrowser(browserPath, cdpPort, pageUrl, userDataDir) {
  const browserArgs = [
    `--remote-debugging-port=${cdpPort}`,
    `--user-data-dir=${userDataDir}`,
    "--no-first-run",
    "--no-default-browser-check",
    pageUrl,
  ];

  return spawn(browserPath, browserArgs, {
    stdio: "ignore",
    detached: true,
  });
}

class WebSocketConnection {
  constructor(wsUrl) {
    this.wsUrl = new URL(wsUrl);
    this.socket = null;
    this.buffer = Buffer.alloc(0);
    this.pending = new Map();
    this.messageId = 0;
    this.handshakeDone = false;
    this.handshakeBuffer = "";
  }

  connect() {
    return new Promise((resolve, reject) => {
      const socket = net.createConnection({
        host: this.wsUrl.hostname,
        port: Number(this.wsUrl.port),
      });
      this.socket = socket;

      socket.on("connect", () => {
        const key = crypto.randomBytes(16).toString("base64");
        const headers = [
          `GET ${this.wsUrl.pathname}${this.wsUrl.search} HTTP/1.1`,
          `Host: ${this.wsUrl.host}`,
          "Upgrade: websocket",
          "Connection: Upgrade",
          `Sec-WebSocket-Key: ${key}`,
          "Sec-WebSocket-Version: 13",
          "\r\n",
        ];
        socket.write(headers.join("\r\n"));
      });

      socket.on("data", (chunk) => {
        if (!this.handshakeDone) {
          this.handshakeBuffer += chunk.toString("latin1");
          const headerEnd = this.handshakeBuffer.indexOf("\r\n\r\n");
          if (headerEnd === -1) {
            return;
          }
          const rawHeaders = this.handshakeBuffer.slice(0, headerEnd);
          if (!rawHeaders.startsWith("HTTP/1.1 101")) {
            reject(new Error(`WebSocket handshake failed: ${rawHeaders}`));
            return;
          }
          this.handshakeDone = true;
          const remaining = Buffer.from(
            this.handshakeBuffer.slice(headerEnd + 4),
            "latin1",
          );
          this.handshakeBuffer = "";
          if (remaining.length > 0) {
            this.consumeFrames(remaining);
          }
          resolve();
          return;
        }
        this.consumeFrames(chunk);
      });

      socket.on("error", reject);
      socket.on("close", () => {
        for (const [, pending] of this.pending) {
          pending.reject(new Error("WebSocket closed"));
        }
        this.pending.clear();
      });
    });
  }

  consumeFrames(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    while (true) {
      const frame = this.readFrame();
      if (!frame) {
        return;
      }
      if (frame.opcode === 0x1) {
        const message = JSON.parse(frame.payload.toString("utf8"));
        if (message.id && this.pending.has(message.id)) {
          const { resolve } = this.pending.get(message.id);
          this.pending.delete(message.id);
          resolve(message);
        }
      }
    }
  }

  readFrame() {
    if (this.buffer.length < 2) {
      return null;
    }

    const firstByte = this.buffer[0];
    const secondByte = this.buffer[1];
    const opcode = firstByte & 0x0f;
    const masked = (secondByte & 0x80) !== 0;
    let payloadLength = secondByte & 0x7f;
    let offset = 2;

    if (payloadLength === 126) {
      if (this.buffer.length < offset + 2) {
        return null;
      }
      payloadLength = this.buffer.readUInt16BE(offset);
      offset += 2;
    } else if (payloadLength === 127) {
      if (this.buffer.length < offset + 8) {
        return null;
      }
      const high = this.buffer.readUInt32BE(offset);
      const low = this.buffer.readUInt32BE(offset + 4);
      payloadLength = high * 2 ** 32 + low;
      offset += 8;
    }

    const maskLength = masked ? 4 : 0;
    const frameLength = offset + maskLength + payloadLength;
    if (this.buffer.length < frameLength) {
      return null;
    }

    let payload = this.buffer.subarray(offset + maskLength, frameLength);
    if (masked) {
      const mask = this.buffer.subarray(offset, offset + 4);
      const unmasked = Buffer.alloc(payload.length);
      for (let i = 0; i < payload.length; i += 1) {
        unmasked[i] = payload[i] ^ mask[i % 4];
      }
      payload = unmasked;
    }

    this.buffer = this.buffer.subarray(frameLength);
    return { opcode, payload };
  }

  send(payload) {
    const data = Buffer.from(JSON.stringify(payload), "utf8");
    const mask = crypto.randomBytes(4);
    let header;

    if (data.length < 126) {
      header = Buffer.alloc(2);
      header[1] = 0x80 | data.length;
    } else if (data.length < 65536) {
      header = Buffer.alloc(4);
      header[1] = 0x80 | 126;
      header.writeUInt16BE(data.length, 2);
    } else {
      header = Buffer.alloc(10);
      header[1] = 0x80 | 127;
      header.writeUInt32BE(0, 2);
      header.writeUInt32BE(data.length, 6);
    }

    header[0] = 0x81;
    const masked = Buffer.alloc(data.length);
    for (let i = 0; i < data.length; i += 1) {
      masked[i] = data[i] ^ mask[i % 4];
    }
    this.socket.write(Buffer.concat([header, mask, masked]));
  }

  sendCommand(method, params = {}) {
    const id = ++this.messageId;
    const payload = { id, method, params };
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send(payload);
    });
  }

  close() {
    if (this.socket) {
      this.socket.end();
    }
  }
}

async function waitForCdp(cdpPort, timeoutMs) {
  return poll(
    async () => {
      const list = await httpGetJson(`http://127.0.0.1:${cdpPort}/json/list`);
      return Array.isArray(list) ? list : null;
    },
    timeoutMs,
    `CDP endpoint on port ${cdpPort}`,
  );
}

async function findTargetPage(cdpPort, pageUrl, timeoutMs) {
  return poll(
    async () => {
      const pages = await httpGetJson(`http://127.0.0.1:${cdpPort}/json/list`);
      return pages.find(
        (page) =>
          page.type === "page" &&
          typeof page.url === "string" &&
          page.url.startsWith(pageUrl),
      );
    },
    timeoutMs,
    `target page ${pageUrl}`,
  );
}

async function openTargetPage(cdpPort, pageUrl) {
  const encoded = encodeURIComponent(pageUrl);
  return httpRequestJson(`http://127.0.0.1:${cdpPort}/json/new?${encoded}`, "PUT");
}

function buildBrowserExpression({ params, includeEvaluation }) {
  const paramsJson = JSON.stringify(params);
  const shouldEvaluate = includeEvaluation ? "true" : "false";
  return `(() => {
    const evaluator =
      typeof window.evaluateTasks === "function"
        ? window.evaluateTasks
        : typeof window.evaluateTask === "function"
          ? window.evaluateTask
          : null;
    const output = {
      href: window.location.href,
      title: document.title,
      hasGetTasks: typeof window.getTasks === "function",
      hasEvaluateTask: typeof window.evaluateTask === "function",
      hasEvaluateTasks: typeof window.evaluateTasks === "function",
      tasks: typeof window.getTasks === "function" ? window.getTasks() : null,
      evaluationInput: ${paramsJson},
      evaluationRaw: null,
      evaluation: null
    };

    if (${shouldEvaluate} && evaluator) {
      const raw = evaluator(${paramsJson});
      output.evaluationRaw = raw;
      output.evaluation = {
        success: Boolean(raw && raw.success),
        score:
          raw && typeof raw.score === "number"
            ? raw.score
            : raw && raw.success
              ? 100
              : 0,
        message:
          raw && typeof raw.message === "string"
            ? raw.message
            : raw && typeof raw.msg === "string"
              ? raw.msg
              : null
      };
    }

    return output;
  })()`;
}

async function runtimeEvaluate(ws, expression) {
  const response = await ws.sendCommand("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });

  if (response.error) {
    throw new Error(response.error.message || "Runtime.evaluate failed");
  }

  const details = response.result;
  if (!details || !details.result) {
    throw new Error("Runtime.evaluate returned no result");
  }

  if ("value" in details.result) {
    return details.result.value;
  }

  if (details.exceptionDetails) {
    throw new Error(details.exceptionDetails.text || "Browser evaluation error");
  }

  return details.result;
}

async function waitForEnvironmentReady(ws, timeoutMs) {
  return poll(
    async () => {
      const state = await runtimeEvaluate(
        ws,
        `(() => ({
          readyState: document.readyState,
          hasAppStore: typeof window.AppStore !== "undefined",
          hasGetTasks: typeof window.getTasks === "function",
          hasEvaluateTask: typeof window.evaluateTask === "function",
          hasEvaluateTasks: typeof window.evaluateTasks === "function"
        }))()`,
      );
      const ready =
        (state.readyState === "interactive" || state.readyState === "complete") &&
        state.hasAppStore &&
        (state.hasEvaluateTask || state.hasEvaluateTasks);
      return ready ? state : null;
    },
    timeoutMs,
    "environment initialization",
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help || !args.env) {
    console.log(usage());
    process.exit(args.help ? 0 : 1);
  }

  const envId = args.env;
  const envDir = path.join(ROOT, envId);
  const envIndex = path.join(envDir, "index.html");
  if (!fs.existsSync(envIndex)) {
    throw new Error(`Environment not found: ${envId}`);
  }

  const port = Number(args.port || 8787);
  const cdpPort = Number(args["cdp-port"] || 9222);
  const timeoutMs = Number(args["timeout-ms"] || 20000);
  const params = loadParams(args);
  const beforeScript = loadBeforeScript(args);

  const server = await createStaticServer(ROOT, port);
  const pageUrl = `http://127.0.0.1:${port}/${envId}/index.html`;
  let browserProcess = null;
  let ws = null;

  try {
    if (args.browser) {
      const userDataDir =
        args["user-data-dir"] || path.join("/tmp", `scalewob-cdp-${envId}`);
      fs.mkdirSync(userDataDir, { recursive: true });
      browserProcess = launchBrowser(args.browser, cdpPort, pageUrl, userDataDir);
    }

    await waitForCdp(cdpPort, timeoutMs);
    try {
      await openTargetPage(cdpPort, pageUrl);
    } catch (error) {
      // Some browsers reject /json/new if the page is already open. We fall back to polling.
    }
    const targetPage = await findTargetPage(cdpPort, pageUrl, timeoutMs);
    ws = new WebSocketConnection(targetPage.webSocketDebuggerUrl);
    await ws.connect();
    await waitForEnvironmentReady(ws, timeoutMs);

    let beforeResult = null;
    if (beforeScript) {
      beforeResult = await runtimeEvaluate(ws, beforeScript);
    }

    const includeEvaluation = Object.keys(params).length > 0;
    const snapshot = await runtimeEvaluate(
      ws,
      buildBrowserExpression({ params, includeEvaluation }),
    );

    console.log(
      JSON.stringify(
        {
          pageUrl,
          envId,
          beforeApplied: Boolean(beforeScript),
          beforeResult,
          snapshot,
        },
        null,
        2,
      ),
    );
  } finally {
    if (ws) {
      ws.close();
    }
    await new Promise((resolve) => server.close(resolve));
    if (browserProcess && args["close-browser"]) {
      try {
        process.kill(-browserProcess.pid, "SIGTERM");
      } catch (error) {
        // Ignore browser shutdown failures.
      }
    }
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
