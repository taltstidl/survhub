const SVG_NS = "http://www.w3.org/2000/svg";
const PADDING = 8;
const STYLES = {
  text: ["fill", "font-family", "font-size", "font-weight"],
  rect: ["fill", "stroke", "stroke-width"],
  line: ["stroke", "stroke-width", "stroke-dasharray"],
};
const PERCENT_ATTRIBUTES = ["x", "x1", "x2", "width"];

function plainColor(value) {
  const match = value.match(
    /^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.]+))?\)$/,
  );
  if (!match) return value;
  const [r, g, b] = match.slice(1, 4).map((c) => Math.round(c * 255));
  return `rgba(${r}, ${g}, ${b}, ${match[4] ?? 1})`;
}

function contentBounds(svg) {
  const origin = svg.getBoundingClientRect();
  const boxes = [...svg.querySelectorAll("text, line, rect")]
    .filter((element) => !element.closest("clipPath"))
    .map((element) => element.getBoundingClientRect());
  return {
    left: Math.min(...boxes.map((box) => box.left)) - origin.left,
    top: Math.min(...boxes.map((box) => box.top)) - origin.top,
    right: Math.max(...boxes.map((box) => box.right)) - origin.left,
    bottom: Math.max(...boxes.map((box) => box.bottom)) - origin.top,
  };
}

function fontStylesheet() {
  return document.querySelector('link[href*="fonts.bunny.net/css"]')?.href;
}

function dataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

let embeddedFonts;

function fontFaces() {
  embeddedFonts ??= (async () => {
    const href = fontStylesheet();
    if (!href) return "";
    const css = await (await fetch(href)).text();
    const faces = css.match(/@font-face\s*{[^}]*}/g) ?? [];
    const latin = faces.filter((face) => /-latin-\d+-normal\.woff2/.test(face));
    const embedded = await Promise.all(
      latin.map(async (face) => {
        const url = face.match(/url\(([^)]+\.woff2)\)/)[1];
        const font = await dataUrl(await (await fetch(url)).blob());
        return face.replace(/src:[^;]*;/, `src: url(${font}) format("woff2");`);
      }),
    );
    return embedded.join("\n");
  })().catch(() => {
    embeddedFonts = undefined;
    return "";
  });
  return embeddedFonts;
}

function withTheme(theme, callback) {
  const root = document.documentElement;
  const previous = root.dataset.theme;
  root.dataset.theme = theme;
  try {
    callback();
  } finally {
    if (previous === undefined) delete root.dataset.theme;
    else root.dataset.theme = previous;
  }
}

export function exportSvg(svg, { style, scale = 1, theme = "light" } = {}) {
  const { width } = svg.getBoundingClientRect();
  const bounds = contentBounds(svg);
  const clone = svg.cloneNode(true);
  const sources = svg.querySelectorAll("*");

  withTheme(theme, () =>
    clone.querySelectorAll("*").forEach((element, i) => {
      const computed = getComputedStyle(sources[i]);
      for (const name of STYLES[element.tagName] ?? []) {
        const value = plainColor(computed.getPropertyValue(name));
        element.setAttribute(name, value.replace(/(\d)px\b/g, "$1"));
      }
      for (const name of PERCENT_ATTRIBUTES) {
        const value = element.getAttribute(name);
        if (value?.endsWith("%")) {
          element.setAttribute(name, (parseFloat(value) * width) / 100);
        }
      }
      const rotation = new DOMMatrix(computed.transform);
      if (element.tagName === "text" && !rotation.isIdentity) {
        const angle = (Math.atan2(rotation.b, rotation.a) * 180) / Math.PI;
        const x = element.getAttribute("x");
        const y = element.getAttribute("y");
        element.setAttribute("transform", `rotate(${angle} ${x} ${y})`);
      }
    }),
  );

  const outputWidth = Math.ceil(bounds.right - bounds.left + 2 * PADDING);
  const outputHeight = Math.ceil(bounds.bottom - bounds.top + 2 * PADDING);
  const root = document.createElementNS(SVG_NS, "svg");
  root.setAttribute("width", outputWidth * scale);
  root.setAttribute("height", outputHeight * scale);
  root.setAttribute("viewBox", `0 0 ${outputWidth} ${outputHeight}`);

  const fonts =
    style ?? (fontStylesheet() && `@import url("${fontStylesheet()}");`);
  if (fonts) {
    const element = document.createElementNS(SVG_NS, "style");
    element.textContent = fonts;
    root.append(element);
  }

  const content = document.createElementNS(SVG_NS, "g");
  content.setAttribute(
    "transform",
    `translate(${PADDING - bounds.left},${PADDING - bounds.top})`,
  );
  content.append(...clone.childNodes);
  root.append(content);

  const markup = new XMLSerializer().serializeToString(root);
  const namespace = ` xmlns="${SVG_NS}"`;
  const start = markup.indexOf(namespace) + namespace.length;
  return markup.slice(0, start) + markup.slice(start).replaceAll(namespace, "");
}

export async function exportPng(svg, { scale = 3, background = "white" } = {}) {
  const markup = exportSvg(svg, { style: await fontFaces(), scale });
  const url = URL.createObjectURL(
    new Blob([markup], { type: "image/svg+xml" }),
  );
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.fillStyle = background;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0);
    return await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  } finally {
    URL.revokeObjectURL(url);
  }
}
