"use strict";

const byId = (id) => document.getElementById(id);
let latestResult = null;
let busy = false;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}
function safeLink(url, label) {
  const link = node("a", label);
  try {
    const parsed = new URL(url);
    if (["https:", "http:"].includes(parsed.protocol)) link.href = parsed.href;
  } catch { /* Invalid source URLs are rendered as plain text. */ }
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}
function renderAnswer(answer, citations) {
  const root = byId("answer");
  root.replaceChildren();
  const ids = new Set(citations.map((cite) => cite.id));
  for (const line of answer.split("\n").filter(Boolean)) {
    const p = node("p");
    for (const part of line.split(/(\[S\d+\])/g)) {
      const id = part.slice(1, -1);
      if (ids.has(id)) {
        const link = node("button", part, "cite-link");
        link.type = "button";
        link.setAttribute("aria-label", "Перейти к источнику " + id);
        link.addEventListener("click", () => byId("cite-" + id)?.scrollIntoView({behavior:"smooth", block:"center"}));
        p.append(link);
      } else p.append(document.createTextNode(part));
    }
    root.append(p);
  }
}
function renderResult(data) {
  latestResult = data;
  byId("result-question").textContent = data.question;
  byId("provider-badge").textContent = (data.mode === "demo" ? "ДЕМО · " : "LIVE · ") +
    (data.provider === "extractive" ? "Фрагменты" : "LLM");
  byId("time-label").textContent = (data.metrics.latency_ms / 1000).toFixed(2) + " сек.";
  byId("result-state").textContent = data.citations.length ? "Исследование завершено" : "Недостаточно данных";
  renderAnswer(data.answer, data.citations);
  const warnings = byId("warnings");
  warnings.replaceChildren();
  for (const warning of data.warnings) warnings.append(node("div", warning.message, "warning"));
  const citations = byId("citations");
  citations.replaceChildren();
  byId("source-count").textContent = data.citations.length + " фрагментов";
  for (const cite of data.citations) {
    const card = node("article", undefined, "citation-card");
    card.id = "cite-" + cite.id;
    const title = node("div", undefined, "citation-title");
    title.append(node("span", cite.id, "citation-id"), safeLink(cite.url, cite.title + " ↗"));
    const kinds = {pdf:"PDF", abstract:"Аннотация", demo_note:"Авторская демо-заметка · не цитата из статьи"};
    card.append(title, node("div", kinds[cite.kind] + (cite.page ? " · страница " + cite.page : ""), "evidence-kind"), node("blockquote", cite.quote));
    citations.append(card);
  }
  if (!data.citations.length) citations.append(node("p", "Подходящих фрагментов не найдено.", "input-hint"));
  const trace = byId("trace");
  trace.replaceChildren();
  const names = {plan:"План", search:"Поиск", extract:"Извлечение", retrieve:"Retrieval", synthesize:"Ответ"};
  for (const step of data.trace) {
    const li = node("li");
    li.append(node("span", step.status === "ok" ? "✓" : step.status === "partial" ? "◐" : "×", "trace-status " + step.status),
      node("span", names[step.name] || step.name, "trace-name"), node("span", step.detail, "trace-detail"),
      node("span", step.duration_ms + " ms", "trace-time"));
    trace.append(li);
  }
  const metrics = byId("metrics");
  metrics.replaceChildren();
  const grid = node("div", undefined, "metrics-grid");
  const labels = {papers_found:"Статей", pdf_success_rate:"PDF извлечено", citation_id_validity:"Корректность ID", source_coverage:"Покрытие источников", evidence_chunks:"Фрагментов"};
  for (const [key, value] of Object.entries(data.metrics)) {
    if (key === "latency_ms") continue;
    const card = node("div", undefined, "metric");
    const isRate = key.includes("rate") || key.includes("validity") || key.includes("coverage");
    card.append(node("b", isRate ? (value * 100).toFixed(0) + "%" : String(value)),
      node("span", labels[key] || key));
    grid.append(card);
  }
  metrics.append(grid);
  byId("queries").textContent = "Запросы: " + data.queries.join(" / ");
  byId("result").hidden = false;
}
async function runResearch(event) {
  event.preventDefault();
  if (busy) return;
  const sources = [...document.querySelectorAll('input[name="sources"]:checked')].map((item) => item.value);
  if (!sources.length) {
    byId("error").textContent = "Выберите хотя бы один источник поиска.";
    byId("error").hidden = false;
    return;
  }
  busy = true;
  byId("run-button").disabled = true;
  byId("error").hidden = true;
  byId("empty-state").hidden = true;
  byId("result").hidden = true;
  byId("loading").hidden = false;
  byId("result-state").textContent = "Выполняется";
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), 160000);
  try {
    const response = await fetch("/api/research", {
      method:"POST", headers:{"Content-Type":"application/json"}, signal:controller.signal,
      body:JSON.stringify({question:byId("question").value, max_papers:Number(byId("max-papers").value), sources}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Проверьте вопрос и параметры исследования.");
    renderResult(data);
  } catch (error) {
    byId("error").textContent = error.name === "AbortError" ? "Превышено время ожидания. Попробуйте более узкий вопрос." : error.message;
    byId("error").hidden = false;
    byId("result-state").textContent = "Требуется повтор";
    if (latestResult) byId("result").hidden = false;
    else byId("empty-state").hidden = false;
  } finally {
    clearTimeout(deadline);
    busy = false;
    byId("run-button").disabled = false;
    byId("loading").hidden = true;
  }
}
byId("research-form").addEventListener("submit", runResearch);
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => {
  byId("question").value = button.dataset.question;
  byId("question").focus();
}));
byId("download").addEventListener("click", () => {
  if (!latestResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(latestResult, null, 2)], {type:"application/json"}));
  const anchor = node("a");
  anchor.href = url;
  anchor.download = "deepscholar-research.json";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
fetch("/api/config").then((r) => { if (!r.ok) throw new Error("Config unavailable"); return r.json(); }).then((config) => {
  byId("mode-label").textContent = config.mode === "demo" ? "Демо-корпус" : "Живой поиск";
  byId("mode-note").textContent = config.mode === "demo"
    ? "Демо на трёх статьях. Фрагменты — авторские заметки, а не оригинальные аннотации. Поиск в интернете выключен."
    : "Поиск arXiv / OpenAlex. Если PDF недоступен, используем аннотацию с явной пометкой.";
  [...byId("max-papers").options].forEach((option) => { option.disabled = Number(option.value) > config.max_papers; });
}).catch(() => {
  byId("mode-label").textContent = "Сервис недоступен";
  byId("mode-note").textContent = "Не удалось загрузить настройки сервиса. Проверьте backend.";
});
