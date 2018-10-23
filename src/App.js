import React, { Component } from 'react';
import Reveal from 'reveal.js';
import MathJax from 'react-mathjax2';
import C3Chart from 'react-c3js';
import SyntaxHighlighter from 'react-syntax-highlighter';
import { monokaiSublime } from 'react-syntax-highlighter/styles/hljs';
import 'c3/c3.css';
import '../node_modules/reveal.js/css/reveal.css';
import '../node_modules/reveal.js/css/theme/sky.css';
import './App.css';

const MATHJAX_CDN_URL =
  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=AM_HTMLorMML';

const tf = 'tf(t,d) = log(1 + f_(t,d))';
const sigmoidChartConfig = makeFunctionLineChart(
  -4.5,
  4.5,
  0.5,
  v => (1 / (1 + Math.exp(-v))).toFixed(2),
  ''
);

function makeFunctionLineChart(min, max, step, fn, title) {
  const x = [];
  for (let i = min; i <= max; i += step) {
    x.push(i);
  }
  const y = x.map(fn);
  return {
    title: { text: title },
    data: {
      x: 'x',
      columns: [['x', ...x], ['y', ...y]]
    },
    tooltip: {
      show: true,
      format: {
        value: value => {
          return `x: ${value}`;
        }
      }
    }
  };
}

class App extends Component {
  componentDidMount() {
    Reveal.initialize({
      width: '100%',
      height: '100%'
    });

    document.addEventListener('keydown', e => {
      if (e.code === 'ArrowRight' || e.code === 'ArrowDown') {
        // Temp workaround for the wrong chart size
        window.dispatchEvent(new Event('resize'));
      }
    });
  }

  render() {
    return (
      <div className="demo reveal">
        <div className="slides">
          <section>
            <h2>Sentiment Analysis</h2>
            <h3>An application of Naive Bayes and Logistic Regression</h3>
          </section>
          <section>
            <header>MathJax</header>
            <div>
              <a href="http://asciimath.org/">See AsciiMath</a>
            </div>
            <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
              <div>
                <MathJax.Node>{tf}</MathJax.Node>
              </div>
            </MathJax.Context>
          </section>
          <section>
            <header>C3 Chart</header>
            <div>
              <C3Chart className="demo-chart" {...sigmoidChartConfig} />
            </div>
          </section>
          <section>
            <header>Code SyntaxHighlighter</header>
            <SyntaxHighlighter
              language="python"
              style={monokaiSublime}
              wrapLines={true}
            >
              {`
                def sigmoid(self, x):
                  return 1.0 / (1 + np.exp(-x))
              `}
            </SyntaxHighlighter>
          </section>
        </div>
      </div>
    );
  }
}

export default App;
