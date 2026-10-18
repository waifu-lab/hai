// 氫原子：一個質子，外面一個電子。
// 電子雲依 1s 軌域的徑向機率分布 P(r) ∝ r²·e^(−2r/a₀) 取樣，也就是 Gamma(3, a₀/2)，
// 再疊上一條半徑 a₀ 的波耳軌道，上面只有一個電子。

const CENTER = 260;
const BOHR_RADIUS = 78; // a₀，單位是 SVG 座標
const POINT_COUNT = 2600;
const MAX_RADIUS = 230;

// 固定種子的亂數，伺服器和瀏覽器算出同一張圖，避免 hydration 不一致
function mulberry32(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const random = mulberry32(1);

const points = Array.from({ length: POINT_COUNT }, (_, i) => {
  // Gamma(3) = 三個指數分布相加
  const r =
    (-Math.log((1 - random()) * (1 - random()) * (1 - random())) *
      BOHR_RADIUS) /
    2;
  // 球面上均勻的方向
  const cosTheta = 2 * random() - 1;
  const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
  const phi = 2 * Math.PI * random();
  const radius = Math.min(r, MAX_RADIUS);
  const x = radius * sinTheta * Math.cos(phi);
  const y = radius * sinTheta * Math.sin(phi);
  const depth = (cosTheta + 1) / 2; // 0 在後面，1 在前面
  return {
    id: i,
    x: Number((CENTER + x).toFixed(2)),
    y: Number((CENTER + y).toFixed(2)),
    r: Number((0.5 + depth * 0.8).toFixed(2)),
    opacity: Number((0.15 + depth * 0.55).toFixed(2)),
  };
}).sort((a, b) => a.opacity - b.opacity);

export function HydrogenAtom({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`relative aspect-square text-emerald-600 dark:text-emerald-300 ${className ?? ""}`}
    >
      <div className="absolute inset-[25%] rounded-full bg-emerald-500/10 blur-3xl dark:bg-emerald-400/10" />
      <svg viewBox="0 0 520 520" className="relative size-full">
        <g fill="currentColor">
          {points.map((p) => (
            <circle key={p.id} cx={p.x} cy={p.y} r={p.r} opacity={p.opacity} />
          ))}
        </g>
        <circle
          cx={CENTER}
          cy={CENTER}
          r={BOHR_RADIUS}
          fill="none"
          stroke="currentColor"
          strokeOpacity={0.55}
          strokeWidth={1}
        />
        <g className="origin-[260px_260px] motion-safe:animate-[spin_24s_linear_infinite]">
          <circle
            cx={CENTER + BOHR_RADIUS}
            cy={CENTER}
            r="12"
            fill="currentColor"
            opacity={0.15}
          />
          <circle
            cx={CENTER + BOHR_RADIUS}
            cy={CENTER}
            r="5"
            fill="currentColor"
          />
        </g>
        <circle
          cx={CENTER}
          cy={CENTER}
          r="18"
          fill="currentColor"
          opacity={0.15}
        />
        <circle cx={CENTER} cy={CENTER} r="8" fill="currentColor" />
      </svg>
      <div className="absolute top-[6%] left-[6%] rounded-md border bg-fd-background/70 px-2.5 py-1.5 font-mono leading-tight backdrop-blur-sm">
        <div className="text-[10px] text-fd-muted-foreground">1</div>
        <div className="text-xl font-semibold text-fd-foreground">H</div>
        <div className="text-[10px] text-fd-muted-foreground">1.008</div>
      </div>
    </div>
  );
}
