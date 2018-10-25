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

const bayesTherom = 'P(A|B) = (P(B|A) * P(A))/(P(B))';
const logitModel = 'h_theta(x) = g(Xtheta) = 1/(1+e^(-Xtheta))';
const tf = 'tf(t,d) = log(1 + f_(t,d))';
const featureVector = 'F = (f_1, f_2, ..., f_n)';
const targetVector = 'C = (C_1, C_2, ..., C_k)';
const featureBayes =
  'obrace(P(C_k|F))^("posterior") = (obrace(P(F|C_k))^("likelihood") * obrace(P(C_k))^("prior"))/(ubrace(P(F))_("marginal likelihood"))';
const jointProb = 'P(F|C_k)*P(C_k) = P(f_1, f_2, f_3, ..., f_n, C_k)';
const jointProb2 =
  ' = P(f_1|f_2, f_3, ..., f_n, C_k)*P(f_2|f_3, ..., f_n, C_k)*...*P(f_n|C_k)*P(C_k)';
const jointProb3 = ' = P(f_1|C_k)*P(f_2|C_k)*...*P(f_n|C_k)*P(C_k)';
const jointProb4 = '= P(C_k)prod_(i=0)^N P(f_i|C_k)';
const bayesClassifier =
  'hat y = argmax P(C_k)prod_(i=0)^N P(f_i|C_k), k in {1, ..., K}';

const sigmoidChartConfig = makeFunctionLineChart(
  -4.5,
  4.5,
  0.5,
  v => (1 / (1 + Math.exp(-v))).toFixed(2),
  ''
);

const accuracyChart = {
  title: { text: 'Model Accuracy' },
  data: {
    columns: [
      ['My Logistic', 87, 72, 70.7],
      ['sklearn Logistic', 89, 82.5, 82.5],
      ['MNB', 86, 80.7, 80.6],
      ['SVM', 91, 81.6, 81.7]
    ],
    type: 'bar'
  },
  bar: {
    width: {
      ratio: 0.5
    }
  },
  axis: {
    x: {
      type: 'category',
      categories: [
        'IMDB Review Bi-gram feature',
        'Twitter Sentiment Bi-gram',
        'Overall Bi-gram'
      ]
    }
  },
  tooltip: {
    show: true,
    format: {
      value: value => {
        return `${value} %`;
      }
    }
  }
};

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
            <h2>
              <strong>Naive Bayes and Logistic Regression</strong>
            </h2>
            <h3>And application on Sentiment Analysis</h3>
          </section>
          <section>
            <div className="demo-text-block">Naive Bayes</div>
            <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
              <div className="demo-text-block--math">
                <MathJax.Node>{bayesTherom}</MathJax.Node>
              </div>
            </MathJax.Context>
            <div className="demo-text-block">Language processing</div>
            <div className="demo-text-block--math">
              tf-idf feature, ngram language model
            </div>
            <div className="demo-text-block">Logistic Regression</div>
            <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
              <div className="demo-text-block--math">
                <MathJax.Node>{logitModel}</MathJax.Node>
              </div>
            </MathJax.Context>
            <div className="demo-text-block">Sentiment Analysis</div>
            <div className="demo-text-block--math">Demo</div>
          </section>
          <section>
            <header>
              <h2>Naive Bayes</h2>
            </header>
            <section>
              <header>
                <h5>Bayes classifier</h5>
              </header>
              <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                <div>
                  <MathJax.Node>{featureVector}</MathJax.Node>
                </div>
              </MathJax.Context>
              <br />
              <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                <div>
                  <MathJax.Node>{targetVector}</MathJax.Node>
                </div>
              </MathJax.Context>
              <br />
              <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                <div>
                  <MathJax.Node>{featureBayes}</MathJax.Node>
                </div>
              </MathJax.Context>
            </section>
            <section>
              <div className="fragment" data-fragment-index="1">
                <div>Joint Probability</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{jointProb}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <div>Probability chain rule</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{jointProb2}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="3">
                <div>Conditional independence assumption</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{jointProb3}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="4">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{jointProb4}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <div className="fragment" data-fragment-index="1">
                <div>Maximum likelihood estimation</div>
                <br />
                <div>
                  <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                    <div>
                      <MathJax.Node>{bayesClassifier}</MathJax.Node>
                    </div>
                  </MathJax.Context>
                </div>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                Generative classifier
              </div>
            </section>
          </section>
          <section>
            <header>
              <h2 />
            </header>
          </section>
          <section>
            <div>Sentiment Analysis example</div>
            <div>
              <C3Chart {...accuracyChart} />
            </div>
            <br />
            <div style={{ 'font-size': '1.2vw' }}>
              <div>
                Training accuracy and speed on the whole data set(IMDB +
                Twitter)
              </div>
              <div>1145k Train cases and 505k Validation cases</div>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy %</th>
                    <th>Training Time</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>My Logistic(10000 iteration)</td>
                    <td>70.7%</td>
                    <td>101 min</td>
                  </tr>
                  <tr>
                    <td>Sklearn Logistic</td>
                    <td>82.5%</td>
                    <td>3.3 min</td>
                  </tr>
                  <tr>
                    <td>Multinomial Naive Bayes</td>
                    <td>80.6%</td>
                    <td>2 min</td>
                  </tr>
                  <tr>
                    <td>Support Vector Machine</td>
                    <td>81.7%</td>
                    <td>2.6 min</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </div>
      </div>
    );
  }
}

export default App;
