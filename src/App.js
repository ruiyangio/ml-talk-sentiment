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
import mrImage from './mr.png';
import derivativeImage from './derivative.gif';
import objFImage from './f.png';
import objD1Image from './1d.png';

const MATHJAX_CDN_URL =
  'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=AM_HTMLorMML';

const bayesTherom = 'P(A|B) = (P(B|A) * P(A))/(P(B))';
const featureVector = 'F = (f_1, f_2, ..., f_n)';
const targetVector = 'C = (C_1, C_2, ..., C_k)';
const featureBayes =
  'obrace(P(C_k|F))^("posterior") = (obrace(P(F|C_k))^("likelihood") * obrace(P(C_k))^("prior"))/(ubrace(P(F))_("marginal likelihood"))';
const jointProb = 'P(F|C_k)*P(C_k) = P(f_1, f_2, f_3, ..., f_n, C_k)';
const jointProb2 =
  ' = P(f_1|f_2, f_3, ..., f_n, C_k)*P(f_2|f_3, ..., f_n, C_k)*...*P(f_n|C_k)*P(C_k)';
const jointProb3 = ' = P(f_1|C_k)*P(f_2|C_k)*...*P(f_n|C_k)*P(C_k)';
const jointProb4 = '= P(C_k)prod_(i=1)^N P(f_i|C_k)';
const bayesClassifier =
  'hat y = argmax P(C_k)prod_(i=0)^N P(f_i|C_k), k in {1, ..., K}';
const wordModel =
  'hatp(w_i|C_k) = (count(w_i, C_k)) /(sum_(winV)count(w, C_k))';
const smoothing =
  'hatp(w_i|C_k) = (count(w_i, C_k) + alpha) /(sum_(winV)count(w, C_k) + alpha|V|)';
const wordClassifier = 'hat y = argmax P(C_k)prod_(i=0)^N P(w_i|C_k)';
const logClassifier =
  'log(haty) = argmax log(P(C_k)) + sum_(i=0)^N log(P(w_i|C_k))';
const linearClassifier = '= beta_0 + W_k^T*X';
const logLikelihood =
  'log(P(w_i|C_k)) = log(count(w_i, C_k) + alpha) - log(sum_(winV)count(w, C_k) + alpha|V|)';
const ngramModelJoint =
  'P(w_1|w_2, w_3, ..., w_n, C_k)*P(w_2|w_3, ..., w_n, C_k)*...*P(w_n|C_k)*P(C_k)';
const ngramModel = ' = P(w_1|w_2, C_k)*P(w_2|w_3, C_k)*...*P(w_n|C_k)*P(C_k)';
const tf = 'tf(t,d) = log(1 + f_(t,d))';
const idf = 'idf(t,D) = log(|D|/ |{d in D: t in d}| + alpha)';
const regression = 'y~~f(X,beta)';
const conditionalExpectation = 'E(Y|X) = f(X,beta)';
const logitModel = 'h_theta(x) = g(Xtheta) = 1/(1+e^(-Xtheta))';
const linearRegression = 'y = Xbeta, yinRR';
const generalizedModel = 'E(Y|X)=f(X,beta)=mu=g^-1(Xbeta)';
const linkFunction = 'Xbeta = g(mu)';
const sentExample = 'P(positive|F) = p, P(negative|F) = 1 - p';
const logOdds = 'ln(p/(1-p)) = Xtheta';
const sigmoid = 'S(x) = e^x / (e^x + 1) = 1 / (1 + e^-x)';
const sigmoidPdfInt =
  'F(x;mu;s) = intf(x;mu;s)dx = inte^(-(x-mu)/s)/(s*(1+e^(-(x-mu)/s))^2)dx = 1/(1 + e^(-(x-mu)/s))';
const sigmoidInt = 'intS(x)dx = ln(1+e^x)';
const sigmoidDerivative =
  "S^'(x) = -(1+e^-x)^-2*(-e^-x) = 1/(1+e^-x)*(1-1/(1+e^-x)) = S(x)*(1-S(x))";
const massFunction1 = 'P(y=1|Xtheta) = h_theta(X)';
const massFunction2 = 'P(y=0|Xtheta) = 1-h_theta(X)';
const massFunction = 'P(y|Xtheta) = h_theta(X)^y*(1-h_theta(X))^(1-y)';
const massFunctionSpread =
  'P(y|Xtheta) = prod_ih_theta(x_i)^(y_i)*(1-h_theta(x_i)^(1-y_i))';
const logProb =
  'll = 1/nsum_i^ny_ilog(h_theta(x_i))+(1-y_i)*log(1-h_theta(x_i))';
const logProb2 =
  '= 1/nsum_i^ny_ilog((h_theta(x_i))/(1-h_theta(x_i)))+log(1-h_theta(x_i))';
const gradientAsc = 'theta_n = theta_(n-1) + alpha*(partialll)/(partialtheta)';
const gradient =
  'gradf(x_1, x_2,...,x_3) = [(partialf)/(partialx_1), (partialf)/(partialx_2), ...,(partialf)/(partialx_n)]';
const logProb3 = '1/nsum_i^ny_iXtheta-log(1+e^(Xtheta))';
const gradientAwd =
  '(partialll)/(partialtheta) = 1/nsumyX - X*(e^(Xtheta)/(1+e^Xtheta))';
const gradientAwd2 = '= X^T(Y-(1/(1+e^-(X*theta))))';

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
                <h5>
                  <strong>Bayes classifier</strong>
                </h5>
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
            <section>
              <h5>
                <strong>Multinomial Naive Bayes for text classification</strong>
              </h5>
              <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                <div>
                  <MathJax.Node>{wordModel}</MathJax.Node>
                </div>
              </MathJax.Context>
              <div className="fragment" data-fragment-index="1">
                <br />
                <div>Add Smoothing factor</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{smoothing}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                Lidstone Smoothing: 0&lt;&alpha;&lt;1
              </div>
              <br />
              <div className="fragment" data-fragment-index="3">
                Laplace Smoothing: &alpha; = 1
              </div>
            </section>
            <section>
              <h4>
                <strong>In log space</strong>
              </h4>
              <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                <div>
                  <MathJax.Node>{wordClassifier}</MathJax.Node>
                </div>
              </MathJax.Context>
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{logClassifier}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{linearClassifier}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="3">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{logLikelihood}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <a
                href="https://github.com/ruiyangio/hadoop-naive-bayes-classifier"
                target="_blank"
                rel="noopener noreferrer"
              >
                MapReduce implementation
              </a>
              <div>
                <img src={mrImage} alt="MR" />
              </div>
              <div>81% accuracy using IMDB review data</div>
            </section>
          </section>
          <section>
            <header>
              <h2>Language Proccessing</h2>
            </header>
            <section>
              <h4>
                <strong>ngram language model</strong>
              </h4>
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{ngramModelJoint}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <div>Markov assumption</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{ngramModel}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="3">
                <SyntaxHighlighter
                  language="python"
                  style={monokaiSublime}
                  wrapLines={true}
                >
                  {`
                    text = "Lucy eats ice cream"
                    bigram = ["Lucy eats", "eats ice",  "ice cream"]
                    grams = ["Lucy", "eats", "ice", "cream", "Lucy eats", "eats ice",  "ice cream"]
                    `}
                </SyntaxHighlighter>
              </div>
            </section>
            <section>
              <h4>
                <strong>TF-IDF as feature</strong>
              </h4>
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{tf}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{idf}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="2" />
              <br />
              <div className="fragment" data-fragment-index="3">
                Alternative is to use stop words, but it's not good for
                Sentiment Analysis
              </div>
            </section>
            <section>
              <h4>
                <strong>Negation</strong>
              </h4>
              <SyntaxHighlighter
                language="python"
                style={monokaiSublime}
                wrapLines={true}
              >
                {`
                  text = "I really don't dislike this cake"
                  negated_grams = ['i', 'really', 'i really', 'do',
                  'really do', "n't", "do n't", 'NOT_dislike', "n't NOT_dislike",
                  'NOT_this', 'NOT_dislike NOT_this', 'NOT_cake', 'NOT_this NOT_cake']
                  `}
              </SyntaxHighlighter>
            </section>
          </section>
          <section>
            <header>
              <h2>Logistic Regression</h2>
            </header>
            <section>
              <h4>
                <strong>Regression Analysis</strong>
              </h4>
              <div>
                A biological phenomenon that the heights of descendants of tall
                ancestors tend to regress down towards a normal average.
              </div>
              <br />
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{regression}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{conditionalExpectation}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <h4>
                <strong>Generalized Linear Model</strong>
              </h4>
              <div>
                Linear combination + transformation function &#10132; prediction
              </div>
              <div className="fragment" data-fragment-index="0">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{linearRegression}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{generalizedModel}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{linkFunction}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="3">
                <div>
                  Y is from a particular distribution in the exponential family
                </div>
              </div>
            </section>
            <section>
              <h5>
                <strong>Logistic Regression</strong>
              </h5>
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{sentExample}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{logOdds}</MathJax.Node>
                  </div>
                </MathJax.Context>
                <br />
                <div>
                  a constant change in the feature leads to a constant change in
                  log odds
                </div>
              </div>
              <br />
              <div className="fragment" data-fragment-index="3">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{logitModel}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <h4>
                <strong>Sigmoid</strong>
              </h4>
              <div className="demo-chart demo-chart__half demo-chart__center">
                <C3Chart {...sigmoidChartConfig} />
              </div>
              <br />
              <div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{sigmoid}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <div>
                <div>
                  Cumulative distribution function of Logistic distribution
                </div>
                <br />
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{sigmoidPdfInt}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="0">
                <div>Softplus function</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{sigmoidInt}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="1">
                <div>Derivative of sigmoid function</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{sigmoidDerivative}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <div className="fragment" data-fragment-index="0">
                <div>
                  <strong>Probability Mass function</strong>
                </div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div className="demo-text-block--math">
                    <MathJax.Node>{massFunction1}</MathJax.Node>
                  </div>
                </MathJax.Context>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div className="demo-text-block--math">
                    <MathJax.Node>{massFunction2}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="1">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div className="demo-text-block--math">
                    <MathJax.Node>{massFunction}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div className="demo-text-block--math">
                    <MathJax.Node>{massFunctionSpread}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="3">
                <div>
                  <strong>Reward function</strong>
                </div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div className="demo-text-block--math">
                    <MathJax.Node>{logProb}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="4">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div className="demo-text-block--math">
                    <MathJax.Node>{logProb2}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <h4>
                <strong>Derivative</strong>
              </h4>
              <div className="fragment" data-fragment-index="0">
                <img
                  src={derivativeImage}
                  alt="derivative"
                  style={{ width: '55%' }}
                />
              </div>
            </section>
            <section>
              <h4>
                <strong>Gradient</strong>
              </h4>
              <div className="fragment" data-fragment-index="0">
                <div>A vector of first order partial derivative</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{gradient}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="1">
                Points to the direction of largest increase
              </div>
            </section>
            <section>
              <div className="fragment" data-fragment-index="0">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{logProb3}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="1">
                <div>Partial derivative</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{gradientAwd}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <div className="fragment" data-fragment-index="2">
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{gradientAwd2}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
              <br />
              <div className="fragment" data-fragment-index="4">
                <div>Iteratively update unknown parameters</div>
                <MathJax.Context input="ascii" script={MATHJAX_CDN_URL}>
                  <div>
                    <MathJax.Node>{gradientAsc}</MathJax.Node>
                  </div>
                </MathJax.Context>
              </div>
            </section>
            <section>
              <div>
                <div>Object function</div>
                <img
                  src={objFImage}
                  alt="object function"
                  className="demo-img__quarter"
                />
              </div>
            </section>
            <section>
              <div>
                <div>Derivative</div>
                <img
                  src={objD1Image}
                  alt="derivative"
                  className="demo-img__quarter"
                />
              </div>
            </section>
            <section>
              <h5>
                <strong>Implementation</strong>
              </h5>
              <SyntaxHighlighter
                language="python"
                style={monokaiSublime}
                wrapLines={true}
              >
                {`
                    def sigmoid(self, x):
                      return 1.0 / (1 + np.exp(-x))
                    
                    def fit(self, x, y):
                        self.w = np.zeros(x.shape[1])
                        n_sample = x.shape[0]
                        for i in range(self.max_iteration):
                            scores = x.dot(self.w)
                            y_pred = self.sigmoid(scores)
                            error = y - y_pred
                            gradient = 1/n_sample*x.T.dot(error)
                            self.w += self.learning_rate * gradient
                    
                    def predict(self, x):
                        scores = x.dot(self.w)
                        return np.round(self.sigmoid(scores))
                    `}
              </SyntaxHighlighter>
              <div>
                <a
                  href="https://github.com/ruiyangio"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Find more on my Github
                </a>
              </div>
            </section>
          </section>
          <section>
            <div>Sentiment Analysis results</div>
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
          <section
            data-background-iframe="http://game.westus2.cloudapp.azure.com/ml/graphql"
            data-background-interactive
          />
        </div>
      </div>
    );
  }
}

export default App;
