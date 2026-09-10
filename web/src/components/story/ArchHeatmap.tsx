// Architecture: which heads sit in the locked copy set, and how joint ablation grows.

import { ARCH, JOINT, PROTOCOL } from '../../data'

const CELL = 36
const GAP = 8
const LABEL = 28

export function ArchHeatmap() {
  const layers = PROTOCOL.nLayers
  const heads = PROTOCOL.nHeads
  const w = LABEL + heads * (CELL + GAP)
  const h = LABEL + layers * (CELL + GAP) + 8

  return (
    <div className="chart-wrap">
      <svg
        className="chart-svg"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={`Attention heads in a ${layers} by ${heads} grid. Named copy heads are filled.`}
      >
        {Array.from({ length: heads }, (_, hi) => (
          <text
            key={`h${hi}`}
            x={LABEL + hi * (CELL + GAP) + CELL / 2}
            y={14}
            textAnchor="middle"
            fontSize="10"
            fill="var(--chart-label)"
            fontFamily="var(--font-mono)"
          >
            h{hi}
          </text>
        ))}
        {Array.from({ length: layers }, (_, li) => (
          <text
            key={`l${li}`}
            x={2}
            y={LABEL + li * (CELL + GAP) + CELL / 2 + 4}
            fontSize="10"
            fill="var(--chart-label)"
            fontFamily="var(--font-mono)"
          >
            L{li}
          </text>
        ))}
        {ARCH.map((cell) => {
          const x = LABEL + cell.head * (CELL + GAP)
          const y = LABEL + cell.layer * (CELL + GAP) - 8
          return (
            <rect
              key={`${cell.layer}-${cell.head}`}
              className="mix-cell"
              x={x}
              y={y}
              width={CELL}
              height={CELL}
              rx="3"
              fill={cell.named ? 'var(--chart-named)' : 'var(--bg-sunk)'}
              stroke={cell.named ? 'var(--chart-named)' : 'var(--chart-dead)'}
            />
          )
        })}
      </svg>
      <ul className="legend">
        <li>
          <span className="swatch" style={{ borderColor: 'var(--chart-named)', background: 'var(--chart-named)' }} />
          in the locked copy set
        </li>
        <li>
          <span className="swatch" style={{ borderColor: 'var(--chart-dead)', background: 'transparent' }} />
          leftover head
        </li>
      </ul>
    </div>
  )
}

export function JointAblation() {
  const w = 340
  const h = 220
  const pad = { l: 44, r: 14, t: 14, b: 40 }
  const plotW = w - pad.l - pad.r
  const plotH = h - pad.t - pad.b
  const nMax = Math.max(...JOINT.map((j) => j.n), 1)
  const y = (v: number) => pad.t + plotH - (v / 100) * plotH
  const bw = plotW / Math.max(nMax, 1) - 10

  return (
    <div className="chart-wrap">
      <svg
        className="chart-svg"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label="Task accuracy drop as more heads are ablated together, in sweep order."
      >
        {[0, 50, 85, 100].map((t) => (
          <g key={t}>
            <line
              x1={pad.l}
              x2={w - pad.r}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--chart-grid)"
              strokeWidth="1"
              strokeDasharray={t === 85 ? '4 3' : undefined}
            />
            <text x={pad.l - 6} y={y(t) + 3} textAnchor="end" fontSize="10" fill="var(--chart-label)" fontFamily="var(--font-mono)">
              {t}
            </text>
          </g>
        ))}
        {JOINT.map((j, i) => {
          const x = pad.l + (j.n - 0.5) * (plotW / nMax) - bw / 2
          const top = y(j.drop)
          const bh = y(0) - top
          return (
            <g key={j.n}>
              <rect
                className="bar-grow"
                x={x}
                y={top}
                width={bw}
                height={Math.max(bh, 0)}
                fill="var(--chart-named)"
                style={{ animationDelay: `${i * 80}ms` }}
              />
              <text x={x + bw / 2} y={h - 8} textAnchor="middle" fontSize="10" fill="var(--chart-label)" fontFamily="var(--font-mono)">
                {j.n}
              </text>
            </g>
          )
        })}
      </svg>
      <p className="meta story-caption" style={{ marginTop: '0.5rem', textTransform: 'none', letterSpacing: 0 }}>
        Joint drop (%) after adding 1..n heads. Single-head zeros often do nothing.
      </p>
    </div>
  )
}
