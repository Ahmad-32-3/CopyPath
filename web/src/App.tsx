import { ArchHeatmap, JointAblation } from './components/story/ArchHeatmap'
import { ResultBento } from './components/story/ResultBento'
import { ResultsBars } from './components/story/ResultsBars'
import { StackGrid } from './components/story/StackGrid'
import { StoryBeat } from './components/story/StoryBeat'
import {
  ABLATION,
  DECISIONS,
  ILLUSTRATIVE,
  METRICS,
  NEXT,
  PROTOCOL,
  SECTORS,
} from './data'

const TOC = [
  { href: '#problem', label: 'The problem' },
  { href: '#answer', label: 'The answer' },
  { href: '#result', label: 'The result' },
  { href: '#stack', label: 'Tech stack' },
  { href: '#decisions', label: 'Design decisions' },
  { href: '#next', label: 'Next and real world' },
  { href: '#close', label: 'Close' },
]

const NOTE = ILLUSTRATIVE
  ? ' These numbers are illustrative until a measured run overwrites data.ts.'
  : ''

const named = PROTOCOL.namedHeads.map(([l, h]) => `L${l}h${h}`).join(', ')

export function App() {
  return (
    <>
      <a className="skip-link" href="#problem">
        Skip to the walkthrough
      </a>

      {ILLUSTRATIVE ? (
        <div className="banner" role="status">
          Illustrative numbers. Do not read these as a trained-net intervention result.
        </div>
      ) : null}

      <div className="masthead">
        <div className="masthead__inner">
          <div className="masthead__mark">
            <b>CopyPath</b> · find the copy trick, then check it by turning pieces off
          </div>
          <ul className="masthead__nav">
            <li>
              <a href="#problem">problem</a>
            </li>
            <li>
              <a href="#result">result</a>
            </li>
            <li>
              <a href="#decisions">decisions</a>
            </li>
            <li>
              <a href="#next">next</a>
            </li>
          </ul>
        </div>
      </div>

      <main className="page">
        <header className="page-hero">
          <p className="meta">Walkthrough · a tiny copy puzzle and checks that should change the answer</p>
          <h1>CopyPath</h1>
          <p className="lead">
            A tiny network can get every answer right on a simple copy puzzle and still leave me unsure
            where it stores that trick. I train a small transformer to copy a token, then I turn off or
            swap the pieces I claimed were doing the copy. If my story is real, those edits should change
            the answer in a predicted way. Random pieces should not. The number I trust is how often
            those checks behave as claimed.
          </p>
          <p className="intro-detail">
            Sequences are length {PROTOCOL.seqLen} over a vocab of {PROTOCOL.vocab}: when the prompt
            looks like [x, y, z, x], the model should emit y, the token that followed the first x.
            Success is how often the checks I named behave the way I claimed: turn those pieces off
            and the copy should break; put them back and it should return.
            {ILLUSTRATIVE
              ? ' Numbers below are labeled illustrative.'
              : ` Measured check rate is ${METRICS.successPct}%.`}
          </p>
          <nav aria-label="On this page">
            <ul className="toc">
              {TOC.map((item) => (
                <li key={item.href}>
                  <a href={item.href}>{item.label}</a>
                </li>
              ))}
            </ul>
          </nav>
        </header>

        <StoryBeat
          id="problem"
          kicker="The problem"
          title="Getting every copy right still does not tell me where the trick lives"
          caption="A perfect score on the puzzle is not a map of which pieces do the copy."
        >
          <p>
            I can train a tiny network until it copies correctly on every held-out sequence I sample.
            That score does not tell me which pieces implement the copy. If I only report puzzle
            accuracy, I have a black box that happens to be small.
          </p>
          <p>
            I am not claiming large language models are solved. I am asking whether I can recover or
            break this one copy trick with the edits I named, better than random edits.
          </p>
        </StoryBeat>

        <StoryBeat
          id="answer"
          kicker="The answer"
          title="Turn off the pieces I named. Put them back. See if the copy follows."
          caption="Corrupt the copied token, write the named pieces back in from a clean run, and score whether the right token returns."
        >
          <p>
            The task is a copy puzzle. The model sees [x, y, z, x] and should emit y, the token after
            the first x. I train a one-layer attention-only net (no extra feed-forward layer) so the
            copy cannot hide there.
          </p>
          <p>
            I sweep every head, then lock the smallest joint set whose removal drops copy accuracy
            by at least 20 points. I predict: that set breaks and restores; a leftover head does
            not. Random heads are a debug column on the same sequences, not the result.
          </p>
        </StoryBeat>

        <StoryBeat
          id="result"
          kicker="The result"
          title={`${METRICS.successPct}% of the named checks behaved as claimed`}
          caption={`Which heads sit in the copy set, and how often named edits beat random edits.${NOTE}`}
          visual={
            <>
              <ResultBento />
              <ArchHeatmap />
              <JointAblation />
              <ResultsBars />
              <div className="compare-table-wrap" style={{ marginTop: '1rem' }}>
                <table className="compare-table">
                  <caption className="sr-only">Intervention success versus random heads and chance</caption>
                  <thead>
                    <tr>
                      <th scope="col">Method</th>
                      <th scope="col">Success %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ABLATION.map((row) => (
                      <tr key={row.model}>
                        <td>{row.label ?? row.model}</td>
                        <td>{row.successPct}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          }
        >
          <p>
            On {PROTOCOL.nEval} sequences the intact net copies at {METRICS.taskPct}%. The locked
            suite on named heads {named || 'the copy set'} hits {METRICS.successPct}%: the checks
            behaved as claimed. Random-head edits on the same sequences score {METRICS.randomPct}%.
            Chance is {METRICS.chancePct}%.
          </p>
          <p>
            One head at a time often does nothing. The copy is spread across a small set. Turning
            that whole set off is what breaks the algorithm; writing the set back in from a clean
            run is what restores it.
          </p>
        </StoryBeat>

        <StoryBeat
          id="stack"
          kicker="Tech stack"
          title="What trains the net and what draws the page"
          caption="Each tool in one plain sentence, then a short technical line."
          visual={<StackGrid />}
        >
          <p>
            Python and PyTorch train the tiny net. Numpy circuits cover the unit tests so a random
            patch cannot fake a restore. Vite builds the page; React draws it; Tailwind styles it;
            motion handles enter animations and counters with reduced-motion respect.
          </p>
        </StoryBeat>

        <StoryBeat
          id="decisions"
          kicker="Design decisions"
          title="Why the check rate comes before another accuracy number"
          caption="Each row is a path I could have taken, and the one I took instead."
        >
          <ul className="decision-list">
            {DECISIONS.map((d) => (
              <li key={d.built}>
                <span className="decision-first">{d.first}</span>
                <span className="decision-built">{d.built}</span>
              </li>
            ))}
          </ul>
        </StoryBeat>

        <StoryBeat
          id="next"
          kicker="Improve · run · real world"
          title="What is weak, how to run it, who can use it today"
          caption="Exact commands below. Sectors are jobs, not a pitch deck."
        >
          <h3>What I would improve</h3>
          <ul>
            {NEXT.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
          <h3>Run on your machine</h3>
          <pre className="code-block">
            <code>{`python -m pytest tests/test_eval.py -q
python scripts/run.py
npm --prefix web install
npm --prefix web run dev`}</code>
          </pre>
          <h3>How this applies today</h3>
          <ul>
            {SECTORS.map((s) => (
              <li key={s.name}>
                <strong>{s.name}.</strong> {s.job}
              </li>
            ))}
          </ul>
        </StoryBeat>

        <section className="story-beat" id="close">
          <div className="story-beat__grid">
            <div className="story-beat__copy">
              <p className="story-kicker">Close</p>
              <h2>I pointed at the copy, not at a chatbot</h2>
              <div className="story-prose">
                <p>
                  On this vocab, this sequence length, and this attention-only net, the named
                  head set restored and broke the copy as claimed. The random column stays visible
                  so the percentage cannot stand alone.
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </>
  )
}
