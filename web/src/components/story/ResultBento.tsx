// The four result numbers, in a bento grid. Each counter rolls up from zero
// when it scrolls into view; before that (or with reduced motion, or a hidden
// tab) it just shows the real value from data.ts. The number on screen is
// always the true one. This is the Kokonut bento's job, done in plain CSS so
// it carries the field-kit palette and skips the tilt and cursor chrome.

import { useEffect, useRef, useState } from 'react'
import { animate, useReducedMotion } from 'motion/react'
import { COUNTERS, type Counter } from '../../data'

function Cell({ c }: { c: Counter }) {
  const reduce = useReducedMotion()
  const [shown, setShown] = useState(c.value)
  const ref = useRef<HTMLDivElement>(null)
  const decimals = Number.isInteger(c.value) ? 0 : c.value < 10 ? 2 : 1

  useEffect(() => {
    if (reduce) return
    const el = ref.current
    if (!el) return
    let controls: ReturnType<typeof animate> | undefined
    const io = new IntersectionObserver(
      (entries) => {
        if (!entries[0].isIntersecting) return
        io.disconnect()
        setShown(0)
        controls = animate(0, c.value, {
          duration: 1,
          ease: [0.22, 1, 0.36, 1],
          onUpdate: (v) => setShown(v),
          onComplete: () => setShown(c.value),
        })
      },
      { threshold: 0.4 },
    )
    io.observe(el)
    return () => {
      io.disconnect()
      controls?.stop()
    }
  }, [c.value, reduce])

  return (
    <div className="bento__cell" ref={ref}>
      <div className="bento__value">
        {shown.toFixed(decimals)}
        {c.unit ? <span className="bento__unit"> {c.unit}</span> : null}
      </div>
      <div className="bento__label">{c.label}</div>
      <div className="bento__note">{c.note}</div>
    </div>
  )
}

export function ResultBento() {
  return (
    <div className="bento" role="group" aria-label="Intervention and task results">
      {COUNTERS.map((c) => (
        <Cell key={c.key} c={c} />
      ))}
    </div>
  )
}
