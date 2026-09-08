/* GLM-5.3 模型验收测试工具前端逻辑 */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));

// 保存服务端返回的关键元信息（用于判断是否已配置真实 Key）
window.__cfgMeta = {
  hasApiKey: false,   // 后端 config.json / 环境变量是否已有真实 api_key
  hasModelId: false,
};

const toastEl = $("#toast");
let toastTimer = null;
function toast(msg, type = "") {
  toastEl.textContent = msg;
  toastEl.className = "toast show " + (type || "");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2500);
}

// ---------- Collapse ----------
$$(".collapse > .collapse-trigger").forEach(tr => {
  tr.addEventListener("click", () => {
    tr.parentElement.classList.toggle("open");
  });
});

// ---------- Config Load/Save ----------
const CFG_MAP = {
  api_base: "#cfg-api-base",
  api_key: "#cfg-api-key",
  model_id: "#cfg-model-id",
  timeout_seconds: "#cfg-timeout",
  max_retries: "#cfg-max-retries",
  public_base_url: "#cfg-public-base",
};
const CAP_MAP = {
  support_thinking: "#cap-thinking",
  thinking_default_enabled: "#cap-thinking-default",
  support_reasoning_effort: "#cap-reasoning-effort",
  support_tool_choice: "#cap-tool-choice",
};
const RUN_MAP = {
  run_functional: "#run-functional",
  run_benchmark: "#run-benchmark",
};
const BENCH_MAP = {
  total_sessions: "#bench-sessions",
  arrival_rate_start: "#bench-arrival-start",
  arrival_rate_end: "#bench-arrival-end",
  ramp_duration_seconds: "#bench-ramp",
  steady_duration_seconds: "#bench-steady",
  init_prompt_length_avg: "#bench-init-len",
  input_length_avg: "#bench-input-len",
  output_length_avg: "#bench-output-len",
};

async function loadConfig() {
  try {
    const r = await fetch("/api/config");
    const c = await r.json();
    // 记住后端是否已有真实 Key / ModelId
    window.__cfgMeta.hasApiKey = Boolean(c.api_key_masked) || Boolean(c.api_key);
    window.__cfgMeta.hasModelId = Boolean(c.model_id);
    // basic
    for (const [k, sel] of Object.entries(CFG_MAP)) {
      if (k === "api_key") {
        // 不回填真实key（保持原样），但遮蔽显示
        const input = $(sel);
        if (c.api_key_masked) {
          input.placeholder = c.api_key_masked + "（留空则不修改）";
        }
        continue;
      }
      const v = c[k];
      if (v !== undefined && v !== null) $(sel).value = v;
    }
    // capability
    if (c.capability) {
      for (const [k, sel] of Object.entries(CAP_MAP)) {
        $(sel).checked = !!c.capability[k];
      }
    }
    // run toggles
    for (const [k, sel] of Object.entries(RUN_MAP)) {
      $(sel).checked = !!c[k];
    }
    // benchmark
    if (c.benchmark) {
      for (const [k, sel] of Object.entries(BENCH_MAP)) {
        const v = c.benchmark[k];
        if (v !== undefined && v !== null) $(sel).value = v;
      }
    }
    // 多模态样本自检告警
    const problems = c.media_sample_problems || [];
    const warnEl = $("#sample-warning");
    if (problems.length) {
      $("#sample-warning-body").textContent = problems.join("；");
      warnEl.style.display = "";
    } else {
      warnEl.style.display = "none";
    }
    updateRunButtonState();
  } catch (e) {
    toast("加载配置失败: " + e.message, "err");
  }
}

function collectConfigPayload() {
  const payload = {};
  for (const [k, sel] of Object.entries(CFG_MAP)) {
    const v = $(sel).value;
    if (k === "api_key") {
      if (v && v.trim()) payload.api_key = v.trim();
      continue;
    }
    if (v !== "" && v !== null && v !== undefined) {
      if (k === "timeout_seconds") payload[k] = parseInt(v, 10) || 300;
      else if (k === "max_retries") payload[k] = parseInt(v, 10) || 0;
      else payload[k] = v;
    }
  }
  // public_base_url 允许清空（留空 = 关闭自托管样本），需显式传空串
  payload.public_base_url = $("#cfg-public-base").value.trim();
  payload.capability = {};
  for (const [k, sel] of Object.entries(CAP_MAP)) {
    payload.capability[k] = $(sel).checked;
  }
  for (const [k, sel] of Object.entries(RUN_MAP)) {
    payload[k] = $(sel).checked;
  }
  payload.benchmark = {};
  for (const [k, sel] of Object.entries(BENCH_MAP)) {
    const v = $(sel).value;
    if (v !== "" && v !== undefined) {
      payload.benchmark[k] = k.includes("rate") || k.includes("length")
        ? parseFloat(v)
        : parseInt(v, 10);
    }
  }
  return payload;
}

function validateRequiredConfigForRun(payload) {
  const issues = [];
  const base = payload.api_base || $("#cfg-api-base").value;
  const model = payload.model_id || $("#cfg-model-id").value || (window.__cfgMeta.hasModelId ? "env" : "");
  // API Key 判定优先级：
  //  1) 输入框本次填入 → payload.api_key
  //  2) 后端已保存（config.json 或 环境变量） → __cfgMeta.hasApiKey
  const keyInput = $("#cfg-api-key").value.trim();
  const hasKey = Boolean(payload.api_key || keyInput || window.__cfgMeta.hasApiKey);
  if (!base) issues.push("API Base");
  if (!model) issues.push("Model ID");
  if (!hasKey) issues.push("API Key");
  return issues;
}

function updateRunButtonState() {
  const payload = collectConfigPayload();
  const issues = validateRequiredConfigForRun(payload);
  const btn = $("#btn-run");
  if (issues.length) {
    btn.disabled = true;
    $("#banner-sub").textContent = `请先配置必填项：${issues.join("、")}`;
  } else {
    btn.disabled = false;
    $("#banner-sub").textContent = "保存配置后点击「开始测试」，工具将逐项执行并自动生成报告。";
  }
}

[CFG_MAP, CAP_MAP, RUN_MAP, BENCH_MAP].forEach(obj => {
  Object.values(obj).forEach(sel => {
    const el = $(sel);
    if (!el) return;
    el.addEventListener("input", updateRunButtonState);
    el.addEventListener("change", updateRunButtonState);
  });
});

$("#btn-save-config").addEventListener("click", async () => {
  const btn = $("#btn-save-config");
  const prev = btn.textContent;
  btn.textContent = "保存中…";
  btn.disabled = true;
  try {
    const payload = collectConfigPayload();
    // 必过项逻辑：support_reasoning_effort 与 capability必过不可取消（给提示但允许用户决定）
    const r = await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error((await r.json()).detail || "保存失败");
    toast("✅ 配置已保存", "ok");
    await loadConfig();
  } catch (e) {
    toast("保存失败: " + e.message, "err");
  } finally {
    btn.textContent = prev;
    btn.disabled = false;
  }
});

$("#btn-reload").addEventListener("click", loadConfig);

// ---------- Run self-test (SSE) ----------
let running = false;
const funcRowMap = new Map();  // id -> <tr>

function ensureFuncTableRows(funcIds = []) {
  const tbody = $("#func-tbody");
  tbody.innerHTML = "";
  for (const id of funcIds) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td class="id-cell">${id}</td>` +
      `<td colspan="8" style="color:var(--text-muted)">等待执行…</td>` +
      `<td></td>`;
    tbody.appendChild(tr);
    funcRowMap.set(id, tr);
  }
  if (!funcIds.length) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-muted);padding:24px">执行后将展示26项结果</td></tr>`;
  }
}

function setPhaseStatus(phase, status) {
  const tab = $(`.phase-tab[data-phase="${phase}"]`);
  if (!tab) return;
  tab.classList.remove("active", "done");
  if (status === "active") tab.classList.add("active");
  if (status === "done") tab.classList.add("done");
}

function consoleAppend(kind, text) {
  const con = $("#console");
  const div = document.createElement("div");
  div.className = "line";
  const ts = new Date().toLocaleTimeString("zh-CN", { hour12: false });
  div.innerHTML = `<span class="ts">${ts}</span>` +
    `<span class="k-${kind}">[${kind.toUpperCase()}]</span> ` +
    `<span>${escapeHtml(text)}</span>`;
  con.appendChild(div);
  con.scrollTop = con.scrollHeight;
}
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderResultRow(row) {
  let tr = funcRowMap.get(row.id);
  if (!tr) {
    tr = document.createElement("tr");
    $("#func-tbody").appendChild(tr);
    funcRowMap.set(row.id, tr);
  }
  const req = row.required
    ? `<span style="color:var(--danger);font-weight:700">是</span>`
    : `<span style="color:var(--text-muted)">否</span>`;
  const statusMap = {
    pass: ["status-pass", "✅ 通过"],
    fail: ["status-fail", "❌ 失败"],
    waive: ["status-waive", "🟡 豁免"],
    error: ["status-error", "💥 异常"],
    skip: ["", "—"],
  };
  const [cls, label] = statusMap[row.status] || ["", row.status];
  const failHtml = (row.failures && row.failures.length)
    ? `<ul class="mini-failures" style="margin:4px 0 0;padding:0;list-style:none">${row.failures.map(f => `<li>• ${escapeHtml(f)}</li>`).join("")}</ul>`
    : "";
  // 诊断说明与失败原因分开呈现：诊断走 notes，不再把 [INFO] 混进红色失败列表
  const noteHtml = (row.notes && row.notes.length)
    ? `<ul style="margin:4px 0 0;padding:0;list-style:none;color:var(--text-muted);font-size:11px">${row.notes.map(n => `<li>· ${escapeHtml(n)}</li>`).join("")}</ul>`
    : "";
  const kindHtml = row.error_kind
    ? `<span class="small-hint" style="color:#b45309">[${escapeHtml(row.error_kind)}]</span> `
    : "";
  const retryHtml = (row.retried && row.status === "pass")
    ? ` <span class="small-hint" style="color:#b45309">(重试后通过)</span>` : "";
  tr.innerHTML = `
    <td class="id-cell">${escapeHtml(row.id)}${retryHtml}</td>
    <td>${escapeHtml(row.category)}</td>
    <td>${escapeHtml(row.name)}</td>
    <td style="text-align:center">${req}</td>
    <td style="text-align:center">${row.pass_rate}%</td>
    <td style="text-align:center">${row.http_status ?? "—"}</td>
    <td style="text-align:center">${row.duration_ms}</td>
    <td style="text-align:center">${row.assertions}</td>
    <td style="text-align:center"><span class="${cls}">${label}</span></td>
    <td>${kindHtml}${failHtml}${noteHtml}</td>
  `;
}

// 当前运行中的会话 id，供取消按钮使用
let currentSid = null;

$("#btn-cancel").addEventListener("click", async () => {
  if (!currentSid) return;
  if (!confirm("确定要取消本轮测试吗？\n已完成的用例结果会保留并生成报告；\n取消后不会执行剩余用例和性能压测（如已勾选）。")) return;
  try {
    const r = await fetch(`/api/cancel/${currentSid}`, { method: "POST" });
    const d = await r.json();
    toast(d.ok ? "已请求取消，将在当前用例结束后停止" : (d.msg || "取消失败"));
  } catch (e) {
    toast("取消失败: " + e.message, "err");
  }
});

$("#btn-run").addEventListener("click", async () => {
  if (running) return;
  // 先保存一次配置（静默），确保服务端用最新值
  try {
    const payload = collectConfigPayload();
    await fetch("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (e) {
    toast("保存配置失败，已取消执行", "err");
    return;
  }

  running = true;
  $("#btn-run").disabled = true;
  $("#btn-run-text").textContent = "测试运行中…";
  $("#btn-run-spinner").style.display = "block";
  $("#btn-cancel").style.display = "";
  $("#progress-panel").classList.add("show");
  $("#report-panel").classList.remove("show");
  funcRowMap.clear();
  ensureFuncTableRows();
  ["functional", "benchmark", "report"].forEach(p => setPhaseStatus(p, ""));
  $("#console").textContent = "";
  $("#progress-subtitle").textContent = "初始化SSE连接…";

  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_override: collectConfigPayload() }),
    });
    if (!resp.ok) {
      const e = await resp.json();
      throw new Error(e.detail || "启动失败");
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    let sid = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop() || "";
      for (const part of parts) {
        if (!part.startsWith("data:")) continue;
        const raw = part.slice(5).trim();
        if (!raw) continue;
        let ev;
        try { ev = JSON.parse(raw); } catch { continue; }
        handleEvent(ev);
      }
    }
  } catch (e) {
    consoleAppend("err", "执行中断: " + e.message);
    toast("运行异常: " + e.message, "err");
  } finally {
    running = false;
    currentSid = null;
    $("#btn-run").disabled = false;
    $("#btn-run-text").textContent = "▶ 再次测试";
    $("#btn-run-spinner").style.display = "none";
    $("#btn-cancel").style.display = "none";
    setTimeout(loadHistory, 500);
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "session_start":
        consoleAppend("info", `会话开始 sid=${ev.sid}`);
        currentSid = ev.sid;
        $("#progress-subtitle").textContent = `会话 ${ev.sid}`;
        break;
      case "phase":
        consoleAppend("info", `【${ev.phase}】${ev.msg}`);
        setPhaseStatus(ev.phase, "active");
        $("#progress-subtitle").textContent = ev.msg;
        break;
      case "phase_done":
        consoleAppend("ok", `【${ev.phase}】完成 · ${ev.msg}`);
        setPhaseStatus(ev.phase, "done");
        break;
      case "phase_skip":
        consoleAppend("warn", `【${ev.phase}】跳过 · ${ev.msg}`);
        setPhaseStatus(ev.phase, "done");
        break;
      case "functional.item": {
        const data = ev.data;
        renderResultRow(data);
        const total = ev.total, done = ev.done;
        $("#progress-subtitle").textContent = `功能测试: ${done}/${total} · 刚完成 ${data.id} ${data.name}`;
        if (data.status === "pass") consoleAppend("ok", `${data.id} ${data.name} → 通过 (${data.duration_ms}ms)`);
        else if (data.status === "fail") consoleAppend("err", `${data.id} ${data.name} → 失败`);
        else if (data.status === "waive") consoleAppend("warn", `${data.id} ${data.name} → 豁免`);
        else if (data.status === "error") consoleAppend("err", `${data.id} ${data.name} → 异常`);
        break;
      }
      case "benchmark.item":
        if (ev.done === 1 || ev.done % 20 === 0) {
          consoleAppend("info", `Benchmark: ${ev.done}/${ev.total || "?"} turns · HTTP=${ev.extra?.http} TTFT=${ev.extra?.ttft}ms lat=${ev.extra?.latency}ms`);
        }
        $("#progress-subtitle").textContent = `性能压测: 已完成 ${ev.done} 个请求轮次`;
        break;
      case "report_gen":
        setPhaseStatus("report", "active");
        consoleAppend("info", ev.msg);
        break;
      case "report_done": {
        setPhaseStatus("report", "done");
        const pass = ev.final_verdict.includes("通过");
        const card = $("#report-verdict");
        card.classList.remove("fail");
        if (!pass) card.classList.add("fail");
        card.classList.add("show");
        $("#report-verdict-text").textContent = ev.final_verdict;
        const fnameOf = (p) => p.split("/").pop();
        const html = $("#download-html");
        const json = $("#download-json");
        const xlsx = $("#download-xlsx");
        const pdf = $("#download-pdf");
        html.href = "/api/download/html/" + fnameOf(ev.paths.html);
        html.style.display = "inline-flex";
        json.href = "/api/download/json/" + fnameOf(ev.paths.json);
        json.style.display = "inline-flex";
        xlsx.href = "/api/download/xlsx/" + fnameOf(ev.paths.xlsx);
        xlsx.style.display = "inline-flex";
        if (pdf && ev.paths.pdf) {
          pdf.href = "/api/download/pdf/" + fnameOf(ev.paths.pdf);
          pdf.style.display = "inline-flex";
        }
        // 某种格式导出失败时不影响其余产物，隐藏对应按钮即可
        for (const [kind, el] of [["html", html], ["json", json], ["xlsx", xlsx], ["pdf", pdf]]) {
          if (el && !ev.paths[kind]) el.style.display = "none";
        }
        $("#report-sub").textContent = "报告文件已生成到本地 output/ 目录"
          + "（每个用例的完整请求/响应/SSE分片见 output/artifacts/）";
        consoleAppend("ok", `报告生成完毕 · 最终结论: ${ev.final_verdict}`);
        if (Array.isArray(ev.reasons)) {
          ev.reasons.forEach(x => consoleAppend("info", "结论依据: " + x));
        }
        $("#progress-subtitle").textContent = "全部执行完成";
        $("#report-panel").classList.add("show");
        break;
      }
      case "report_partial":
        consoleAppend("err", ev.msg);
        toast("部分报告格式导出失败，其余格式仍可下载", "err");
        break;
      case "error":
        consoleAppend("err", ev.msg);
        if (ev.trace) consoleAppend("err", ev.trace.slice(0, 500));
        break;
      case "session_end":
        consoleAppend("info", `会话 ${ev.sid} 结束`);
        break;
    }
  }
});

// ---------- History ----------
async function loadHistory() {
  try {
    const r = await fetch("/api/reports");
    const list = await r.json();
    const host = $("#history-list");
    if (!list.length) {
      host.innerHTML = `<div style="color:var(--text-muted);padding:12px">暂无历史报告，完成一次测试后会出现在这里。</div>`;
      return;
    }
    host.innerHTML = "";
    for (const item of list) {
      const row = document.createElement("div");
      row.className = "history-item";
      const key = item.key.replace("kimi-k3-report-", "");
      const links = [];
      if (item.files.html) links.push(`<a class="btn btn-outline" target="_blank" href="/api/download/html/${item.files.html}">HTML</a>`);
      if (item.files.json) links.push(`<a class="btn btn-outline" href="/api/download/json/${item.files.json}">JSON</a>`);
      if (item.files.xlsx) links.push(`<a class="btn btn-success" href="/api/download/xlsx/${item.files.xlsx}">Excel</a>`);
      if (item.files.pdf) links.push(`<a class="btn btn-outline" href="/api/download/pdf/${item.files.pdf}">PDF</a>`);
      row.innerHTML = `<div><span class="name">报告时间: ${key}</span><span class="ts">（自动保存在 output/ 目录）</span></div>` +
        `<div class="btn-row">${links.join("")}</div>`;
      host.appendChild(row);
    }
  } catch (e) {
    toast("历史列表加载失败: " + e.message, "err");
  }
}

// ---------- Boot ----------
document.addEventListener("DOMContentLoaded", () => {
  loadConfig();
  loadHistory();

  // 性能压测参数：一键回到默认值
  const BENCH_DEFAULTS = {
    "bench-sessions": 30,
    "bench-arrival-start": 0.08,
    "bench-arrival-end": 1.0,
    "bench-ramp": 600,
    "bench-steady": 300,
    "bench-init-len": 28569,
    "bench-input-len": 1353,
    "bench-output-len": 346,
  };
  $("#bench-reset").addEventListener("click", () => {
    for (const [id, val] of Object.entries(BENCH_DEFAULTS)) {
      const el = document.getElementById(id);
      if (el) el.value = val;
    }
    toast("已恢复为文档默认推荐值", "");
  });
});
