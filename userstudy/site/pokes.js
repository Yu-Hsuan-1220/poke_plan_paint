// Draw pokes (normalized yx in [-1, 1] on the 576x320 frame: [y0, x0, y1, x1]) as arrows into an <svg>.
const POKE_W = 576, POKE_H = 320;
const POKE_COLORS = ["#ff3b30", "#ffcc00", "#34c759", "#0a84ff", "#bf5af2", "#ff9f0a", "#64d2ff", "#ff375f"];

function drawPokes(svg, pokes) {
  const NS = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  pokes.forEach((p, k) => {
    const color = POKE_COLORS[k % POKE_COLORS.length];
    const x0 = (p[1] + 1) / 2 * (POKE_W - 1), y0 = (p[0] + 1) / 2 * (POKE_H - 1);
    const x1 = (p[3] + 1) / 2 * (POKE_W - 1), y1 = (p[2] + 1) / 2 * (POKE_H - 1);
    const len = Math.hypot(x1 - x0, y1 - y0);
    const g = document.createElementNS(NS, "g");
    if (len > 6) {
      // arrowhead drawn as a polygon so it can carry the same dark outline as the shaft
      const ux = (x1 - x0) / len, uy = (y1 - y0) / len, hl = Math.min(14, len * 0.6), hw = hl * 0.6;
      const bx = x1 - ux * hl, by = y1 - uy * hl;
      const head = `${x1},${y1} ${bx - uy * hw},${by + ux * hw} ${bx + uy * hw},${by - ux * hw}`;
      for (const [stroke, width] of [["rgba(0,0,0,.65)", 6.5], [color, 3.5]]) {
        const l = document.createElementNS(NS, "line");
        Object.entries({x1: x0, y1: y0, x2: bx, y2: by, stroke, "stroke-width": width, "stroke-linecap": "round"})
          .forEach(([a, v]) => l.setAttribute(a, v));
        g.appendChild(l);
      }
      const h = document.createElementNS(NS, "polygon");
      Object.entries({points: head, fill: color, stroke: "rgba(0,0,0,.65)", "stroke-width": 1.5, "stroke-linejoin": "round"})
        .forEach(([a, v]) => h.setAttribute(a, v));
      g.appendChild(h);
    } else {
      // (almost) no motion: a ring = "this point should stay"
      const r = document.createElementNS(NS, "circle");
      Object.entries({cx: x0, cy: y0, r: 9, fill: "none", stroke: color, "stroke-width": 3})
        .forEach(([a, v]) => r.setAttribute(a, v));
      g.appendChild(r);
    }
    const c = document.createElementNS(NS, "circle");
    Object.entries({cx: x0, cy: y0, r: 5, fill: color, stroke: "#000", "stroke-width": 1.5})
      .forEach(([a, v]) => c.setAttribute(a, v));
    g.appendChild(c);
    svg.appendChild(g);
  });
}
