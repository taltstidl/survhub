import { max } from "d3-array";
import { easePolyOut } from "d3-ease";
import { scaleBand, scaleLinear } from "d3-scale";
import { select } from "d3-selection";
import "d3-transition";
import { exportPng, exportSvg } from "./export.js";

let chartCount = 0;

function barChart(
  container,
  rows,
  {
    title,
    subtitle,
    yLabel,
    height = 300,
    radius = 6,
    whisker = 0.15,
    stagger = 120,
    duration = 700,
  } = {},
) {
  const x = scaleBand()
    .domain(rows.map((d) => d.label))
    .range([0, 100])
    .padding(0.35);
  const y = scaleLinear()
    .domain([0, max(rows, (d) => d.value + d.error[1])])
    .nice()
    .range([height, 0]);

  const pct = (v) => `${v}%`;
  const center = (d) => x(d.label) + x.bandwidth() / 2;
  const lower = (d) => y(d.value - d.error[0]);
  const upper = (d) => y(d.value + d.error[1]);
  const clipId = `bars-clip-${chartCount++}`;

  const svg = select(container).append("svg").attr("height", height);

  const { marginTop, marginLeft } = getComputedStyle(svg.node());
  const header = svg
    .append("g")
    .attr("class", "chart-header")
    .attr(
      "transform",
      `translate(${-parseFloat(marginLeft)},${-parseFloat(marginTop)})`,
    );
  header.append("text").attr("class", "chart-title").attr("y", 20).text(title);
  header
    .append("text")
    .attr("class", "chart-subtitle")
    .attr("y", 44)
    .text(subtitle);

  svg
    .append("clipPath")
    .attr("id", clipId)
    .append("rect")
    .attr("y", -radius)
    .attr("width", "100%")
    .attr("height", height + radius);

  const grid = svg
    .append("g")
    .attr("class", "grid")
    .selectAll("g")
    .data(y.ticks(5))
    .join("g");
  grid
    .append("line")
    .attr("x2", "100%")
    .attr("y1", (t) => y(t))
    .attr("y2", (t) => y(t));
  grid
    .append("text")
    .attr("x", -9)
    .attr("y", (t) => y(t))
    .attr("text-anchor", "end")
    .attr("dominant-baseline", "central")
    .text(y.tickFormat(5));

  svg
    .append("g")
    .attr(
      "transform",
      `translate(${6 - parseFloat(marginLeft)},${height / 2}) rotate(-90)`,
    )
    .append("text")
    .attr("class", "y-label")
    .attr("text-anchor", "middle")
    .attr("dominant-baseline", "central")
    .text(yLabel);

  const top = (d) => y(d.value);
  const barHeight = (d) => height - y(d.value) + radius;
  const bars = svg
    .append("g")
    .attr("clip-path", `url(#${clipId})`)
    .selectAll("rect")
    .data(rows)
    .join("rect")
    .attr("class", (d) => `bar bar-${d.type}`)
    .attr("x", (d) => pct(x(d.label)))
    .attr("width", pct(x.bandwidth()))
    .attr("rx", radius)
    .attr("y", top)
    .attr("height", barHeight);

  const half = x.bandwidth() * whisker;
  const errorBars = svg
    .append("g")
    .selectAll("g")
    .data(rows)
    .join("g")
    .attr("class", "error-bar");
  errorBars
    .append("line")
    .attr("x1", (d) => pct(center(d)))
    .attr("x2", (d) => pct(center(d)))
    .attr("y1", lower)
    .attr("y2", upper);
  for (const bound of [lower, upper]) {
    errorBars
      .append("line")
      .attr("x1", (d) => pct(center(d) - half))
      .attr("x2", (d) => pct(center(d) + half))
      .attr("y1", bound)
      .attr("y2", bound);
  }

  const xAxis = svg.append("g").attr("class", "x-axis");
  xAxis.append("line").attr("x2", "100%").attr("y1", height).attr("y2", height);
  const ticks = xAxis.selectAll("g").data(rows).join("g");
  ticks
    .append("line")
    .attr("x1", (d) => pct(center(d)))
    .attr("x2", (d) => pct(center(d)))
    .attr("y1", height)
    .attr("y2", height + 6);
  ticks
    .append("text")
    .attr("x", (d) => pct(center(d)))
    .attr("y", height + 12)
    .attr("text-anchor", "end")
    .attr("dominant-baseline", "central")
    .text((d) => d.label);

  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  bars
    .attr("y", height)
    .attr("height", radius)
    .transition()
    .delay((d, i) => i * stagger)
    .duration(duration)
    .ease(easePolyOut.exponent(5))
    .attr("y", top)
    .attr("height", barHeight);

  errorBars
    .attr("opacity", 0)
    .transition()
    .delay((d, i) => i * stagger + duration)
    .duration(300)
    .attr("opacity", 1);
}

const leaderboard = fetch(
  new URL("api/leaderboard.json", import.meta.url),
).then((response) => response.json());

async function drawMetric(chart, input) {
  const task = (await leaderboard)[chart.dataset.task];
  const { elo, elo_ci } = task[input.value];
  const content = chart.querySelector(".chart-content");
  content.querySelector(":scope > svg")?.remove();
  barChart(
    content,
    task.labels.map((label, i) => ({
      label,
      type: task.types[i],
      value: elo[i],
      error: elo_ci[i],
    })),
    {
      title: "SurvHub Leaderboard",
      yLabel: "Elo rating",
      subtitle: `${chart.dataset.scenario} scenario using ${input.labels[0].textContent.trim()}`,
    },
  );
}

const themeToggle = document.querySelector(".theme-toggle");
const isDark = () => document.documentElement.dataset.theme === "dark";
themeToggle.setAttribute("aria-pressed", isDark());
themeToggle.addEventListener("click", () => {
  const theme = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  themeToggle.setAttribute("aria-pressed", isDark());
  try {
    localStorage.setItem("theme", theme);
  } catch {}
});

function download(filename, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url));
}

const clone = (id) => document.getElementById(id).content.cloneNode(true);

for (const chart of document.querySelectorAll(".chart[data-task]")) {
  const metrics = chart.closest(".leaderboard").querySelector(".radio-cards");
  const selected = () => metrics.querySelector("input:checked");

  chart.append(clone("chart-chrome"));
  const select = metrics.previousElementSibling.querySelector("select");
  select.setAttribute("aria-label", metrics.getAttribute("aria-label"));
  for (const input of metrics.querySelectorAll("input")) {
    select.add(new Option(input.labels[0].textContent.trim(), input.value));
  }
  select.value = selected().value;
  select.addEventListener("change", () => {
    const input = metrics.querySelector(`input[value="${select.value}"]`);
    input.checked = true;
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
  metrics.addEventListener("change", () => {
    select.value = selected().value;
  });

  new IntersectionObserver(
    ([entry], observer) => {
      if (!entry.isIntersecting) return;
      observer.disconnect();
      drawMetric(chart, selected());
      metrics.addEventListener("change", () => drawMetric(chart, selected()));
    },
    { threshold: 0.5 },
  ).observe(chart);
  for (const button of chart.querySelectorAll("[data-format]")) {
    button.addEventListener("click", async () => {
      const svg = chart.querySelector(".chart-content > svg");
      if (!svg) return;
      const { format } = button.dataset;
      const blob =
        format === "png"
          ? await exportPng(svg)
          : new Blob([exportSvg(svg)], { type: "image/svg+xml" });
      download(
        `survhub-${chart.dataset.task}-${selected().value}.${format}`,
        blob,
      );
    });
  }
}
