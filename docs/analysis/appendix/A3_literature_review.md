# Leap: Literature Review and Frank Assessment
## Short-horizon single-pair FX prediction (Transformer/TFT) + PPO trading — is the objective realistic, and what has demonstrated results?

Date: 2026-09-16. Sources were verified by web search and, where possible, by fetching the abstract page or extracting text from the paper PDF. Items marked "[not independently fetched]" rest on the search-result snippet or on citation in another verified paper.

---

## 0. Executive summary (one page)

1. **Single-pair, intraday FX direction from technical indicators alone is, by the weight of the evidence, an unrealistic primary objective.** Fifty years of work since Meese & Rogoff (1983) show exchange rates at horizons ≤12 months are almost impossible to beat against a random walk; Rossi's (2013) JEL survey concludes predictability is fragile and, where it exists, comes from *linear, low-parameter* models on *macro* predictors, not from technical features. Intraday technical-rule studies on FX (Neely & Weller 2003; Curcio et al. 1997; Kozhan & Salmon 2010, all summarised in Neely & Weller's 2011 St. Louis Fed handbook chapter) find **no positive excess returns once realistic spreads are charged**, and whatever profits existed vanished as algorithmic trading grew (Neely, Weller & Ulrich 2009).

2. **The best deep-learning trading research in the world (Oxford-Man Institute) reports weak FX results even at daily horizon on a diversified portfolio.** In the Momentum Transformer paper (Wood et al. 2022, a TFT-based architecture) the *individual FX futures* average Sharpe is 0.28 at zero cost and turns **negative at 1.5 bps of round-trip cost**; the whole edge comes from diversification across ~50 futures in four asset classes. Zhang, Zohren & Roberts (2020) report that long-only and sign-momentum are *negative* Sharpe in FX and that DRL's FX-sector Sharpe (0.55) is the weakest of the profitable sectors.

3. **Complex sequence models rarely beat simple baselines on noisy series.** Zeng et al. (AAAI 2023) show a one-layer linear model (DLinear) beats Transformer forecasters on standard benchmarks; the M4 competition found pure ML worse than statistical methods, while M5 was won by gradient-boosted trees (LightGBM) rather than deep nets. Gu, Kelly & Xiu (2020) found that even with 900+ predictors and 30 years of panel data the monthly out-of-sample R² peaks at **0.40%**; trees and shallow nets tied. Tens of thousands of bars of a single series is far less data than any of these.

4. **PPO-style DRL on top of a noisy forecaster compounds the problem.** Critical surveys (Millea 2021; Hambly, Xu & Yang 2023) and the FinRL group's own overfitting paper (Gort et al. 2022) document non-reproducibility, backtest overfitting and sensitivity to reward design; Mohammadshafie et al. (2024) document that several DRL algorithms converge to passive "hold" behaviour — exactly Leap's zero-trade degeneracy. The one DRL paper with credible results (ZZR 2020) uses **volatility-scaled reward, 50 diversified futures, daily bars, and additive PnL**, none of which Leap has.

5. **Transaction-cost arithmetic alone rules out most of Leap's design space.** With EURUSD annualised volatility ≈7%, expected absolute move is ≈39 pips/day, ≈8 pips/hour, ≈4 pips/15 min. A round-trip cost of 1 pip (standard retail spread 0.7–1.0 pip; ECN ≈0.1 pip + $3–4/lot ≈ 0.4–0.5 pip all-in) requires a **break-even directional accuracy of 51.3% (daily), 56.3% (H1), 62.7% (M15)** if you trade every bar. The best *published* daily FX directional accuracies are 55–58% (Guyard & Deriaz 2024) on a single year and are almost certainly optimistic. Nobody has credibly published 60%+ on M15 EURUSD.

6. **What has demonstrated, replicated, out-of-sample results:** diversified time-series momentum with volatility targeting (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2012/2017: Sharpe ≈1.0 net of costs & fees over 1903–2012, ≈0.4 as a conservative forward assumption); Deep Momentum Networks (Lim, Zohren & Roberts 2019: Sharpe-loss LSTM that *outputs position size*, ≈2× TSMOM gross, robust to 2–3 bps); cross-sectional FX momentum/carry across 20–48 currencies (Menkhoff et al. 2012); gradient-boosted trees/ensembles on tabular features (Gu-Kelly-Xiu 2020; Krauss, Do & Huck 2017); Markov regime switching (Hamilton 1989; Ang & Timmermann 2012) used for **risk scaling**, not signal generation; and, above all, validation protocols (López de Prado 2018; Bailey et al. 2014/2017; Harvey & Liu 2015; White 2000; Hansen 2005) that would have flagged most of Leap's backtests as noise.

7. **Realistic target:** a single-pair, intraday, technical-only system should be expected to have **net Sharpe ≈0 (often negative)**. A daily-bar, volatility-targeted trend/carry system on **one** pair: net Sharpe roughly 0–0.3, with multi-year drawdowns. A diversified 20–50 instrument daily momentum + carry book with vol targeting: net Sharpe 0.4–1.0 in the literature (before fees). If Leap must stay single-pair on MT5, the honest goal is "do not lose money after costs, and be provably better than random by DSR/SPA test", not "beat the market".

---

## 1. Ranked source list (15 sources)

Ranking is by relevance to the Leap decision (predictability × methodology × actionable guidance).

### 1. López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley. ISBN 978-1-119-48208-6.
- **Link:** https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086 ; Ch.1 on SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104847 ; companion "The 10 Reasons Most Machine Learning Funds Fail" (JPM 2018) https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf
- **Methodology:** Research manual: information-driven bars, fractional differentiation (stationarity with memory), triple-barrier labelling (profit-take / stop / time-out), meta-labelling (secondary model decides *whether* and *how much* to act on a primary signal), sample-weighting for label concurrency, purged & embargoed k-fold CV, combinatorial purged CV (CPCV) generating many backtest paths, PBO / deflated Sharpe, feature importance (MDI/MDA/SFI), bet sizing.
- **Dataset:** Methodological; examples on E-mini futures tick data.
- **Reported result:** N/A (methods). Central claim: standard k-fold and walk-forward on overlapping labels leak information and produce false discoveries; "research by individuals" on single strategies is structurally prone to overfitting.
- **Relevance to Leap:** Leap uses fixed-horizon next-bar labels, presumably random or chronological splits, and ~50–100 indicators of the *same* price series (high multicollinearity, many redundant features). AFML addresses every one of these.
- **Adopt:** triple-barrier labels with volatility-scaled barriers; meta-labelling instead of RL for the act/size decision; purged walk-forward + CPCV; fractional differentiation of price-level features; MDA feature importance to prune indicators.
- **Limitations:** Prescriptive rather than empirical; some techniques (e.g., structural breaks, entropy features) are data-hungry; written for institutional teams with tick data.

### 2. Rossi, B. (2013). "Exchange Rate Predictability." *Journal of Economic Literature* 51(4): 1063–1119. DOI 10.1257/jel.51.4.1063
- **Link:** https://www.aeaweb.org/articles?id=10.1257%2Fjel.51.4.1063 ; working paper PDF https://crei.cat/wp-content/uploads/users/working-papers/Rossi_ExchangeRatePredictability_Feb_13.pdf
- **Methodology:** Comprehensive survey plus own out-of-sample tests of the exchange-rate forecasting literature since Meese & Rogoff (1983): predictors (interest differentials, PPP, monetary fundamentals, Taylor-rule fundamentals, net foreign assets), horizons, models (linear, nonlinear, time-varying), evaluation methods.
- **Dataset:** Major bilateral USD rates, monthly/quarterly, post-Bretton-Woods.
- **Reported result:** "It depends" — on predictor, horizon, sample, model and test. Predictability appears mainly with Taylor-rule or net-foreign-asset predictors, **linear** models, and **few parameters**; the random walk remains very hard to beat out of sample, especially at short horizons.
- **Relevance:** Directly answers "is short-horizon FX direction forecastable?" for the *most-studied* pairs: barely, and not via flexible nonlinear models on price history.
- **Adopt:** Treat the random walk (and a no-trade policy) as the mandatory benchmark; add macro/carry predictors (interest differentials) if predictability is the goal; prefer low-parameter models.
- **Limitations:** Monthly/quarterly horizons; predates modern ML.

### 3. Meese, R. A. & Rogoff, K. (1983). "Empirical exchange rate models of the seventies: Do they fit out of sample?" *Journal of International Economics* 14(1–2): 3–24. DOI 10.1016/0022-1996(83)90017-X
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/002219968390017X ; PDF https://scholar.harvard.edu/files/rogoff/files/51_jie1983.pdf
- **Methodology:** Out-of-sample RMSE comparison of structural monetary models (Frenkel-Bilson, Dornbusch-Frankel, Hooper-Morton) and time-series models vs. a driftless random walk, 1–12-month horizons.
- **Dataset:** USD/GBP, USD/DEM, USD/JPY, trade-weighted USD, 1970s–early 1980s.
- **Reported result:** The random walk forecasts as well as or better than every model at 1–12 months, *even with realised future fundamentals plugged in*.
- **Relevance:** The origin of the "FX is a random walk" prior; any FX model must first show it beats this benchmark on a held-out period with a formal test.
- **Adopt:** Random-walk / zero-return baseline in every evaluation; report Diebold-Mariano or Clark-West tests against it.
- **Limitations:** Monthly data; macro predictors only.

### 4. Neely, C. J. & Weller, P. A. (2011). "Technical Analysis in the Foreign Exchange Market." Federal Reserve Bank of St. Louis WP 2011-001; published as Ch.12 of *Handbook of Exchange Rates* (Wiley, 2012). Summarises Curcio, Goodhart, Guillaume & Payne (1997, *Int. J. Finance & Economics* 2: 267–280), Neely & Weller (2003, *J. Int. Money & Finance* 22: 223–237), Kozhan & Salmon (2010), and Neely, Weller & Ulrich (2009, *JFQA* 44(2)).
- **Link:** https://files.stlouisfed.org/files/htdocs/wp/2011/2011-001.pdf ; Wiley chapter https://onlinelibrary.wiley.com/doi/abs/10.1002/9781118445785.ch12 ; NWU 2009 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1403345
- **Methodology:** Literature review with emphasis on true out-of-sample tests; includes genetic-programming rules on half-hourly data (Neely & Weller 2003), tick-data GA rules (Kozhan & Salmon 2010), support/resistance rules on intraday data (Curcio et al. 1997).
- **Dataset:** Major USD pairs; daily since 1970s; intraday 1996 (half-hourly), 2003 and 2008 (tick, Reuters D3000).
- **Reported result (verbatim from extracted text):** "Neely and Weller (2003) use half-hourly data from 1996 ... Once reasonable transaction costs are taken into account and trade is restricted to times of normal market activity, they find no evidence of positive excess returns. Kozhan and Salmon (2010) ... find that a trading rule based on a genetic algorithm can earn significant profits net of transaction costs in 2003 but that these profits disappear by 2008," attributed to the rise of algorithmic trading (Chaboud et al. 2009: algo share of EUR and JPY volume went from ≈0 to 60%+ over 2003–07). Daily filter/MA rule profits of the 1970s–80s were genuine but "had disappeared by the early 1990s" (NWU 2009).
- **Relevance:** This is the closest published analogue to Leap (intraday, technical features, learned rules, EUR/USD-type pairs). Conclusion: no net profits after costs, and any edge decays.
- **Adopt:** Use it to set the prior (≈0 net edge) and the evaluation standard (true out-of-sample, costs charged, normal-hours only).
- **Limitations:** Pre-2010 data; rules simpler than modern ML; no deep learning.

### 5. Wood, K., Giegerich, S., Roberts, S. & Zohren, S. (2022). "Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture." arXiv:2112.08534 (also *Risk.net* Cutting Edge).
- **Link:** https://arxiv.org/abs/2112.08534 ; code https://github.com/kieranjwood/trading-momentum-transformer
- **Methodology:** Decoder-only **Temporal Fusion Transformer** (VSN → LSTM → interpretable multi-head attention → FFN) trained *directly on the Sharpe ratio* to output volatility-scaled positions (a Deep Momentum Network); inputs are normalised returns and MACD indicators at multiple look-backs; optional change-point-detection features. Expanding-window walk-forward, 5 reruns per experiment.
- **Dataset:** 50 most liquid continuous futures (commodities, equity indices, fixed income, **FX**), Pinnacle CLC, 1990–2020; test 1995–2020.
- **Reported result (extracted from PDF, Exhibits 3 & 10):** Portfolio Sharpe 1995–2020 (gross): Long-only 0.51, TSMOM 1.03, LSTM-DMN 1.70, Decoder-Only TFT 2.54. **Individual-asset Sharpe averaged by class, 2015–2020, vs. round-trip cost C:** TFT FX = 0.28 @0 bps, 0.18 @0.5, 0.08 @1.0, **−0.02 @1.5, −0.32 @3.0 bps**; LSTM FX = 0.11 @0 → −0.46 @3 bps; vanilla Transformer FX = −0.17 even at 0 bps ("perform poorly on FX where they need to be quicker"). Portfolio Sharpe falls from 1.71 to −0.37 as C goes 0→3 bps for the TFT.
- **Relevance:** This is the *same model family Leap uses*, built by the strongest group in the field, and FX is its **weakest** asset class with a single-asset edge that a 1-pip spread (≈0.9 bps) erases. The Sharpe comes from cross-asset diversification and from optimising the trading objective directly, not from forecasting next-bar returns.
- **Adopt:** If a deep net is kept, use the DMN recipe — Sharpe/utility loss, direct position output, vol-scaling, turnover regulariser — on **many** instruments, daily bars; drop the forecast→RL pipeline.
- **Limitations:** Futures not spot retail FX; 0–3 bps costs are institutional; five reruns show notable seed variance ("more complex" TFT less stable).

### 6. Zhang, Z., Zohren, S. & Roberts, S. (2020). "Deep Reinforcement Learning for Trading." *Journal of Financial Data Science* 2(2): 25–40. arXiv:1911.10107
- **Link:** https://arxiv.org/abs/1911.10107 ; https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/06/Deep-Reinforcement-Learning-for-Trading.pdf
- **Methodology:** DQN, policy gradient and A2C with discrete {−1,0,+1} and continuous actions; **reward = volatility-scaled additive PnL minus proportional cost (bp × price × |Δposition|)**; states are normalised returns and MACD/RSI-type features; per-contract volatility targeting; baselines Long, Sign(R) TSMOM, MACD.
- **Dataset:** 50 most liquid futures 2011–2019 (commodity, equity index, fixed income, FX), daily.
- **Reported result (extracted, Table 2, portfolio-level vol targeting):** All-contract Sharpe: Long 0.06, Sign(R) 0.44, MACD 0.09, **DQN 1.29, A2C 1.05, PG 0.75**. **FX sector:** Long −0.35, Sign(R) −0.31, MACD 0.01, DQN 0.55, PG 0.26, A2C 0.33. Text: "Time series strategies work well in trending markets, like fixed income markets, however suffer losses in FX markets where directional moves are less usual." DQN/A2C remain positive up to **25 bp** cost on the diversified portfolio (≈$3.5/contract).
- **Relevance:** The credible DRL-for-trading reference. Its success ingredients — vol-scaled reward, additive (not log-equity) PnL, many contracts, daily bars, simple baselines — are the opposite of Leap's log-equity single-pair intraday PPO. FX is the hardest sector even here.
- **Adopt:** If RL is retained: vol-scaled additive reward, explicit cost term in reward, direct return features (not 100 indicators), evaluation vs. Sign(R)/MACD baselines by sector.
- **Limitations:** 8-year test; futures; no significance tests; A2C over-trades (authors note turnover issue).

### 7. Gu, S., Kelly, B. & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies* 33(5): 2223–2273. DOI 10.1093/rfs/hhaa009
- **Link:** https://academic.oup.com/rfs/article/33/5/2223/5758276 ; NBER w25398 https://www.nber.org/papers/w25398 ; PDF https://dachxiu.chicagobooth.edu/download/ML.pdf
- **Methodology:** OLS, PLS, PCR, elastic net, GLM (splines), random forest, GBRT, NN1–NN5; 94 stock characteristics × 8 macro predictors (~920 features); 18-yr train / 12-yr validation / 30-yr rolling OOS (1987–2016); Diebold-Mariano tests; decile long-short portfolios.
- **Dataset:** ~30,000 US stocks, monthly, 1957–2016 (panel of millions of stock-months).
- **Reported result (extracted):** Benchmark 3-factor OLS OOS R² = **0.16%/month**; OLS with 900+ predictors goes deeply negative; PLS/PCR 0.26–0.27%; **RF 0.33%, GBRT 0.34%, NN1 0.33%, NN3 peak 0.40%**; deeper nets (NN4–5) do *not* improve. Long-short decile Sharpe 1.35 (VW) / 2.45 (EW) for NN; market-timing Sharpe 0.77 vs 0.51 buy-and-hold. "All methods agree on the same set of dominant predictive signals which includes variations on momentum, liquidity, and volatility."
- **Relevance:** Best-case ML predictability with vastly more data than Leap: R² < 0.5% monthly; trees ≈ shallow nets; depth does not help; economic value came from a *cross-section* of thousands of assets, not from one series.
- **Adopt:** GBRT/RF as the baseline model family; shallow nets only if they beat GBRT under DM tests; expect R² measured in tenths of a percent and build the decision layer accordingly.
- **Limitations:** US equities, monthly; pre-cost portfolio results; large-scale panel unavailable to a single-pair system.

### 8. Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023). "Are Transformers Effective for Time Series Forecasting?" *AAAI-23*; arXiv:2205.13504. DOI 10.1609/aaai.v37i9.26317
- **Link:** https://arxiv.org/abs/2205.13504 ; https://dl.acm.org/doi/10.1609/aaai.v37i9.26317
- **Methodology:** Compares Informer, Autoformer, FEDformer, Pyraformer, LogTrans against **DLinear/NLinear** (one-layer linear on trend/remainder decomposition) on 9 long-horizon benchmarks; ablations on look-back length and permutation of inputs.
- **Dataset:** ETT, Electricity, Traffic, Weather, ILI, **Exchange-Rate** (8 daily FX series), etc.
- **Reported result:** DLinear beats or matches all Transformer variants "in most cases by a large margin"; Transformer gains stem from direct multi-step output, not attention's temporal modelling; Transformer errors barely change when inputs are shuffled (they are not exploiting order). On Exchange-Rate the linear model is among the best and all models are near a naive persistence forecast.
- **Relevance:** Undercuts the premise that a TFT should extract more from ~50k bars of EURUSD than a linear or GBM model. Supports "TFT only if it beats DLinear/GBM under a proper test."
- **Adopt:** DLinear/NLinear and naive-persistence baselines in Leap's forecaster comparison; report MASE/pinball loss relative to persistence.
- **Limitations:** Long-horizon point/quantile forecasting benchmarks, not trading PnL; subsequent PatchTST/iTransformer narrowed the gap on some datasets (but not on Exchange-Rate to any economically meaningful degree).

### 9. Lim, B., Zohren, S. & Roberts, S. (2019). "Enhancing Time Series Momentum Strategies Using Deep Neural Networks." *Journal of Financial Data Science* 1(4). arXiv:1904.04912
- **Link:** https://arxiv.org/abs/1904.04912 ; https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3369195
- **Methodology:** Deep Momentum Networks: MLP/WaveNet/LSTM that **output the position directly**, trained with a Sharpe-ratio loss inside the TSMOM volatility-scaling framework (position × σ_target/σ_t); compared with Sharpe-loss vs. MSE/classification losses.
- **Dataset:** 88 continuous futures, 1990–2015, daily.
- **Reported result:** Sharpe-optimised LSTM improves TSMOM by >2× gross; still outperforms with costs up to **2–3 bps**; MSE-trained forecasters translated into positions underperform the Sharpe-loss models.
- **Relevance:** Shows that (a) *learning the trading objective directly* beats *forecast-then-trade*, (b) vol scaling is the core of robustness, (c) the edge survives only a few bps — well below retail FX spreads when turnover is high.
- **Adopt:** Replace "TFT quantile forecast → PPO" with a single Sharpe/utility-loss model producing a bounded position, vol-scaled; or, in the non-deep version, a rules-based TSMOM with vol targeting.
- **Limitations:** Institutional cost levels; futures; pre-2015 sample.

### 10. Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). "Time Series Momentum." *Journal of Financial Economics* 104(2): 228–250. DOI 10.1016/j.jfineco.2011.11.003 — and Hurst, Ooi & Pedersen (2012 AQR white paper; JPM 44(1) 2017), "A Century of Evidence on Trend-Following Investing."
- **Link:** MOP https://www.sciencedirect.com/science/article/pii/S0304405X11002613 ; PDF https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf ; HOP SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026 ; 2012 PDF https://www.chesler.us/resources/academia/A_Century_of_Evidence_on_Trend_Following.pdf
- **Methodology:** Sign of past 1–12-month excess return → long/short each instrument, sized to a constant ex-ante volatility (40%/σ in MOP; 10% portfolio vol target in HOP); equal risk across instruments; subtract estimated transaction costs (HOP) and 2/20 fees.
- **Dataset:** MOP: 58 liquid futures/forwards (equity indices, **currencies**, commodities, bonds), 1965–2009. HOP: 67 markets incl. 12 currency pairs, 1880/1903–2012 (extended to 2016 in the JPM version).
- **Reported result (extracted from HOP 2012 PDF, Exhibit 1):** Jan 1903–Jun 2012: 20.0% gross-of-fee return, 14.3% net of 2/20, 9.9% vol, **Sharpe 1.00 net of fees and simulated costs**, correlation ≈−0.05 to S&P 500 and bonds; positive in every decade. Authors' "very conservative" forward assumption: **Sharpe 0.4 net of fees and costs**. MOP: TSMOM significant in every one of 58 instruments, 1–12-month persistence then reversal; performs best in extreme markets.
- **Relevance:** The most-replicated systematic strategy in the literature; FX is one of its weakest sleeves individually, the strategy works through diversification and volatility targeting. Sets the realistic *ceiling* for a rules-based system: Sharpe ≈0.4–1.0 on 50+ markets.
- **Adopt:** Vol-targeted position sizing; multi-instrument universe (MT5 typically offers 30–60 FX pairs, metals, indices, energies as CFDs); daily/weekly rebalancing; 1–12-month look-backs.
- **Limitations:** Hypothetical backtests by a manager (AQR); crowding has lowered live Sharpe since 2009; costs assumed at institutional levels.

### 11. Menkhoff, L., Sarno, L., Schmeling, M. & Schrimpf, A. (2012). "Currency Momentum Strategies." *Journal of Financial Economics* 106(3): 660–684. DOI 10.1016/j.jfineco.2012.06.009
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0304405X12001353 ; SSRN https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1988679
- **Methodology:** Cross-sectional momentum portfolios (sort 48 currencies on past 1–12-month returns), long winners / short losers; bid-ask-adjusted returns; risk-factor and limits-to-arbitrage analysis.
- **Dataset:** Up to 48 currencies vs USD, monthly spot & forward, 1976–2010.
- **Reported result:** Winner–loser spread up to ~10% p.a.; **"fairly sensitive to transaction costs"** — bid-ask adjustment lowers profitability significantly because momentum loads on high-spread currencies; returns concentrated in minor currencies, high idiosyncratic volatility, not explained by standard risk factors.
- **Relevance:** Shows where FX predictability actually lives (cross-section of many currencies, monthly horizon, carry/momentum) and that even there costs eat much of it.
- **Adopt:** Cross-sectional ranking across the pairs MT5 offers; carry (swap) as a feature; monthly holding.
- **Limitations:** Monthly; needs a broad currency universe; small-country pairs have wide retail spreads.

### 12. Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2017). "The Probability of Backtest Overfitting." *Journal of Computational Finance* 20(4): 39–69 (SSRN 2013). — and Bailey & López de Prado (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management* 40(5): 94–107. — and Bailey et al. (2014) "Pseudo-Mathematics and Financial Charlatanism," *Notices of the AMS* 61(5).
- **Link:** PBO https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253 ; DSR https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551 ; Pseudo-Math https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659 ; Python https://github.com/esvhd/pypbo
- **Methodology:** CSCV: split the matrix of (trials × time) into S blocks, form all C(S, S/2) train/test combinations, measure how often the in-sample best trial under-performs the OOS median → PBO. DSR: expected maximum Sharpe among N trials under the null, E[max SR] ≈ σ_SR·[(1−γ)Z⁻¹(1−1/N) + γZ⁻¹(1−1/(N·e))], then test the observed SR against it with skew/kurtosis-adjusted standard error.
- **Dataset:** Analytical + simulations; shows that with ~5 years of daily data one needs only a handful of trials to find an in-sample Sharpe > 1 by chance.
- **Reported result:** Minimum backtest length needed grows with log of trials; typical strategy-search workflows have PBO well above 0.5.
- **Relevance:** Leap's hyper-parameter sweeps, feature sets and reward penalties are all "trials". Without PBO/DSR, any positive backtest is uninterpretable; with them, most will be revealed as noise.
- **Adopt:** Log every configuration tried; compute DSR against the number of trials; compute PBO via CSCV over the walk-forward paths (or CPCV).
- **Limitations:** Requires disciplined trial logging; DSR assumes trials' Sharpe variance is estimable.

### 13. Harvey, C. R. & Liu, Y. (2015). "Backtesting." *Journal of Portfolio Management* 42(1): 13–28; with Harvey, Liu & Zhu (2016) "…and the Cross-Section of Expected Returns," *RFS* 29(1): 5–68; plus White (2000) "A Reality Check for Data Snooping," *Econometrica* 68(5): 1097–1126, and Hansen (2005) "A Test for Superior Predictive Ability," *JBES* 23(4): 365–380.
- **Link:** Harvey-Liu PDF https://people.duke.edu/~charvey/Research/Published_Papers/P120_Backtesting.PDF ; code https://people.duke.edu/~charvey/backtesting/ ; HLZ https://academic.oup.com/rfs/article/29/1/5/1843824 ; White https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152 ; Hansen https://www.tandfonline.com/doi/abs/10.1198/073500105000000063
- **Methodology:** Multiple-testing haircuts (Bonferroni, Holm, BHY) mapping a reported Sharpe and number of trials to a "haircut Sharpe"; HLZ: a new factor needs **t > 3.0** given ≥316 published factors. White RC / Hansen SPA: bootstrap test that the *best* of many rules beats a benchmark, correcting for the search (Hansen's studentised version is more powerful and less sensitive to poor alternatives).
- **Dataset:** Equity factor literature; White's original application was to technical trading rules on the S&P 500.
- **Reported result:** Realistic haircuts often exceed 50% of the reported Sharpe; a t-stat of 2 is meaningless after a search over dozens of rules.
- **Relevance:** Gives a ready-made test for "does the best Leap configuration beat buy-and-hold / zero / TSMOM after accounting for the search?"
- **Adopt:** Hansen SPA test over the set of tried configurations vs. a no-trade and a TSMOM benchmark; require t > 3 equivalent.
- **Limitations:** Needs the full set of trials (including discarded ones).

### 14. Millea, A. (2021). "Deep Reinforcement Learning for Trading — A Critical Survey." *Data* 6(11): 119. DOI 10.3390/data6110119 — with Hambly, B., Xu, R. & Yang, H. (2023) "Recent Advances in Reinforcement Learning in Finance," *Mathematical Finance* 33(3): 437–503; Gort et al. (2022) "Deep RL for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting," arXiv:2209.05559; Mohammadshafie et al. (2024) "DRL Strategies in Finance: Insights into Asset Holding, Trading Behavior, and Purchase Diversity," arXiv:2407.09557; Liu et al. (2021) FinRL, ACM ICAIF, arXiv:2111.09395.
- **Link:** https://www.mdpi.com/2306-5729/6/11/119 ; https://onlinelibrary.wiley.com/doi/full/10.1111/mafi.12382 ; https://arxiv.org/abs/2209.05559 ; https://arxiv.org/html/2407.09557v1 ; https://arxiv.org/abs/2111.09395
- **Methodology:** Surveys of DRL trading; Gort et al. apply hypothesis testing (PBO) to reject over-fitted agents before deployment; Mohammadshafie et al. compare DDPG, PPO, TD3, SAC, A2C trading behaviour on FinRL stock environments.
- **Dataset:** Mostly equities/crypto; daily.
- **Reported result:** Millea (abstract): "lack of consistency in the community can significantly impede research"; state representation and reward design dominate outcomes; risk measures should enter the reward. Gort et al.: "Existing works ... optimistically reported increased profits in backtesting, which may suffer from the false positive issue due to overfitting"; less-overfitted agents outperform. Mohammadshafie et al.: DDPG/TD3/A2C "display a propensity to remain stationary for extended periods" (conservative holders) while PPO/SAC over-trade; **A2C had the best cumulative reward despite passivity**, i.e., "do nothing" was competitive.
- **Relevance:** Documents exactly Leap's symptoms — degenerate hold policies, seed instability, reward-shaping sensitivity, and backtest-overfit "successes" that vanish. FinRL's own authors now filter agents by overfitting probability.
- **Adopt:** If RL is kept, treat it as a *sizing/execution* layer on a pre-validated signal, with vol-scaled dense reward, ensembles over seeds, and PBO screening; compare to no-trade and TSMOM baselines. Otherwise drop RL (recommended).
- **Limitations:** Surveys; the behaviour paper uses a stock environment with a small action budget.

### 15. Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). "The M4 Competition: 100,000 time series and 61 forecasting methods." *IJF* 36(1): 54–74 — and (2022) "M5 accuracy competition: Results, findings, and conclusions," *IJF* 38(4). — plus Lim, Arık, Loeff & Pfister (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting," *IJF* 37(4): 1748–1764, arXiv:1912.09363; Sezer, Gudelek & Ozbayoglu (2020) "Financial time series forecasting with deep learning: A systematic literature review 2005–2019," *Applied Soft Computing* 90: 106181; Fischer & Krauss (2018) *EJOR* 270(2): 654–669; Krauss, Do & Huck (2017) *EJOR* 259(2): 689–702.
- **Link:** M4 https://www.sciencedirect.com/science/article/pii/S0169207019301128 ; M4 findings https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785 ; TFT https://arxiv.org/abs/1912.09363 ; Sezer https://arxiv.org/abs/1911.13288 ; Fischer-Krauss https://www.sciencedirect.com/science/article/abs/pii/S0377221717310652 ; Krauss et al. https://www.sciencedirect.com/science/article/abs/pii/S0377221716308657
- **Methodology / result:** M4: pure ML methods (MLP, RNN) ranked below classical statistical methods; winners were hybrids/ensembles. M5 (retail sales, rich hierarchical covariates): top entries were **LightGBM ensembles**. TFT: designed for multi-horizon forecasting with static/known/observed covariates on large multi-series datasets (electricity, traffic, retail, volatility); quantile loss; it was *not* validated as a trading signal. Sezer et al.: RNN/LSTM dominate the finance-DL literature, but the review notes lack of common benchmarks and of cost-adjusted evaluation. Fischer & Krauss: LSTM on S&P 500 constituents 1992–2015, 0.46%/day and Sharpe 5.8 *before costs*, but **"as of 2010, excess returns seem to have been arbitraged away with LSTM profitability fluctuating around zero after transaction costs."** Krauss et al. 2017: DNN+GBT+RF ensemble 0.45%/day pre-cost, same decay; GBT/RF individually rivalled the DNN.
- **Relevance:** (i) TFT's home turf is many correlated series with covariates — not one noisy return series; (ii) trees win on tabular features; (iii) even the celebrated LSTM stock results were pre-cost and died after 2010.
- **Adopt:** LightGBM/XGBoost on tabular features as the primary model class; report results net of costs by sub-period; treat TFT as an optional challenger.
- **Limitations:** M-competitions are not financial; Fischer-Krauss is equities cross-section.

**Additional verified supporting sources (not ranked):** Guyard & Deriaz (2024) "Predicting Foreign Exchange EUR/USD direction using machine learning," arXiv:2409.04471 / ACM MLMI 2024 — best **58.52%** daily accuracy, one test year (2022), no cost analysis in abstract; López-Herrera, González Maiz Jiménez & Reyes Santiago (2025) "Directional forecasting for eight forex pairs against the US dollar using machine learning techniques," *Discover Artificial Intelligence* — logistic regression with a directional loss beat RF/XGBoost/NN, results "highly sensitive to transaction costs", VaR models badly mis-calibrated (29% violations vs 5% expected); Hamilton (1989) *Econometrica* 57: 357–384 and Ang & Timmermann (2012) *Annual Review of Financial Economics* 4: 313–337 (regime switching captures fat tails, volatility clustering and changing correlations — useful for risk scaling; regime *forecasting* remains hard); retail spread data 2026 from broker-comparison sites (standard accounts 0.7–1.0 pip EURUSD, raw/ECN ≈0.1 pip + $3–4 per lot round-turn; spreads widen at news and outside London/NY overlap) — practitioner, not peer-reviewed: https://forexspreadcompare.com/artigos/eurusd-spread-comparison , https://tradingbeasts.com/what-is-the-average-spread-on-eurusd/

---

## 2. Comparison table

| Approach dimension | Leap's current approach | Research approach with evidence | Expected improvement |
|---|---|---|---|
| **Universe** | One spot pair (EURUSD-style), single OHLCV series, ~10⁴–10⁵ bars | 50–88 futures / 20–48 currencies; edge arises from diversification (MOP 2012; HOP 2012/17; Menkhoff 2012; ZZR 2020; Wood 2022) | Single-asset FX Sharpe in the best papers ≈0.1–0.3 gross; diversified portfolio 1–2.5 gross, ≈0.4–1.0 net. Moving to 20+ MT5 instruments is the single largest available gain. |
| **Horizon / bar** | H1/M15 next-bar forecasts | Daily bars, 1–12-month look-backs, monthly-to-weekly rebalancing (MOP; Menkhoff; Lim 2019); intraday technical rules show no net profit (Neely & Weller 2003; Curcio 1997; Kozhan & Salmon 2010) | Break-even accuracy falls from 56–63% (H1/M15 at 1 pip) to ≈51% (daily). Turnover drops 20–100×, so costs stop dominating. |
| **Features** | 50–100 technical indicators from the same price series | Normalised returns at several look-backs + MACD-type trend signals, volatility, carry/interest differential (ZZR; Lim; Wood; Rossi; Menkhoff); fractional differentiation; MDA importance pruning (AFML) | Lower variance, less collinearity, fewer "trials"; carry is the only FX predictor with robust economic backing. |
| **Labels / target** | Next-bar price/return regression, quantiles | Triple-barrier labels with vol-scaled barriers; meta-labels (AFML); or no labels at all — learn position directly with Sharpe loss (Lim 2019; Wood 2022) | Labels match how PnL is actually realised (stop/target), remove look-ahead in fixed-horizon labels, and let a second model filter false positives. |
| **Forecast model** | TFT with quantile outputs | Baseline ladder: persistence / zero → DLinear (Zeng 2023) → logistic/ridge (López-Herrera 2025; Rossi 2013) → LightGBM/XGBoost (M5; GKX 2020: trees ≈ NN, depth doesn't help) → TFT only if it wins DM/SPA tests | Expect R² in tenths of a percent at best; GBM gives equal accuracy with 100× less tuning surface; avoids the TFT seed-variance Wood et al. report. |
| **Decision policy** | PPO, discrete buy/sell/hold, reward = log-equity + penalties | Volatility-targeted position = signal × σ_target/σ_t, capped (MOP; HOP; Lim; ZZR); if learned, Sharpe/utility-loss network with turnover regulariser; if RL, vol-scaled *additive* reward with explicit cost term and daily bars (ZZR) | Removes the zero-trade degenerate equilibrium (constant-risk sizing means "flat" is not the default), removes reward-scale non-stationarity, and gives a monotone signal→position mapping that can be audited. |
| **Costs in training/eval** | Penalties in RL reward; spread realism unclear | Explicit round-trip cost in bps in reward/loss and in every reported metric; sensitivity curves 0–3 bps (Wood; Lim) or 0–25 bp (ZZR); normal-hours-only trading (Neely & Weller 2003) | Prevents the "profitable at 0 bps, negative at 1.5 bps" surprise that Wood et al. document for FX. |
| **Validation** | Train/test split, backtest, paper trade | Purged/embargoed walk-forward; CPCV; PBO via CSCV; Deflated Sharpe with trial count; Hansen SPA vs. no-trade and TSMOM; t > 3 hurdle (AFML; Bailey et al.; Harvey & Liu; HLZ; White; Hansen) | Converts "unstable results" into a measurable false-discovery rate; most current configurations will fail, which is the correct outcome. |
| **Regime handling** | Online learning / regime detection to adapt models | Hamilton (1989) two-state Markov switching on volatility for **risk scaling**; change-point features as inputs (Wood 2022, +17% Sharpe 2015–20); avoid re-training the forecaster on the fly (adds trials, invites leakage) | Stable, interpretable volatility regime → position cap; avoids chasing noise with online updates. |
| **Success metric** | Equity curve / return | Net Sharpe with DSR p-value, PBO, drawdown, turnover, cost share of gross PnL, vs. baselines by sub-period (Fischer-Krauss post-2010 decay) | Makes the objective testable and stops rewarding luck. |

---

## 3. Frank assessment: is "short-horizon single-pair FX direction from technical indicators + PPO" realistic?

**No — not as a profit objective.** The reasoning, with citations:

1. **The prior is a random walk.** Meese & Rogoff (1983) and the 30-year survey by Rossi (2013) establish that even models fed *realised future fundamentals* fail to beat a random walk at 1–12 months; where predictability exists it comes from linear, low-parameter models with macro predictors. Technical features on one price series are the weakest possible predictor set in this literature.

2. **The intraday technical evidence is directly negative.** Neely & Weller (2003) on half-hourly data: "no evidence of positive excess returns" after costs; Curcio et al. (1997) intraday support/resistance: no reliable profits; Kozhan & Salmon (2010): tick-data GA rules profitable in 2003, gone by 2008 as algorithmic volume rose to 60%+ (Chaboud et al. 2009). Neely, Weller & Ulrich (2009) show every documented FX rule's profit decays after discovery. A retail MT5 system in 2026 competes against exactly those algorithms.

3. **Even the best deep-learning trading research finds FX the hardest sector.** Wood et al. (2022): individual FX futures Sharpe for a TFT-based Momentum Transformer = 0.28 gross, ≈0 at 1.5 bps; vanilla Transformers negative in FX even gross. ZZR (2020): TSMOM sign rule −0.31 Sharpe in FX, DQN 0.55 — the weakest of the profitable sectors — and the headline 1.29 Sharpe needed 50 contracts. Lim et al. (2019) edge survives only to 2–3 bps. These are daily bars on futures; Leap is intraday spot at retail spreads.

4. **Cost arithmetic (own calculation; assumptions stated).** EURUSD annualised vol ≈7% (range 5–9% in recent years), price 1.10 → 1 pip ≈ 0.9 bps. Expected absolute move E|r| = σ√(2/π):
   - Daily: σ ≈ 48 pips, E|r| ≈ 39 pips → break-even accuracy p* = 0.5 + c/(2E|r|) = **50.6% / 51.3% / 51.9%** for round-trip cost 0.5 / 1.0 / 1.5 pips.
   - H1: σ ≈ 10 pips, E|r| ≈ 8 pips → **53.2% / 56.3% / 59.5%**.
   - M15: σ ≈ 5 pips, E|r| ≈ 4 pips → **56.3% / 62.7% / 69.0%**.
   Standard retail spreads are 0.7–1.0 pips; ECN all-in ≈0.4–0.5 pips; both widen at news. Published *daily* FX directional accuracies top out at 55–58% on single test years (Guyard & Deriaz 2024) and are almost certainly inflated by selection; Fischer & Krauss's celebrated LSTM sat near 50–54% and its post-cost profit went to zero after 2010. **Nothing in the literature credibly delivers the 57–63% needed on H1/M15 EURUSD.** (Holding positions for several bars lowers the per-bar cost share but then the relevant horizon is effectively daily and the "next-bar forecaster" is irrelevant.)

5. **Data volume.** Gu-Kelly-Xiu needed millions of stock-months and 900+ features to reach 0.40% monthly R², and deeper nets did not help. TFT (Lim et al. 2021) was validated on tens of thousands of *related* series with covariates. Leap has one series with ~50k autocorrelated bars: too little to fit a TFT meaningfully, but plenty to overfit it (Bailey et al. 2014: a few trials on 5 years of daily data yield a spurious Sharpe > 1).

6. **RL adds a second unstable learner on top of a near-zero-signal input.** PPO must learn from a reward whose signal-to-noise is set by the forecaster; the log-equity reward is dominated by market variance, not by action quality (ZZR explain why they use vol-scaled *additive* PnL). Mohammadshafie et al. (2024) find passive holding is a natural attractor and *competitive* in cumulative reward; Gort et al. (2022) and Millea (2021) document that reported DRL profits are mostly overfitting or irreproducible. A zero-trade policy is, in an honest environment with costs, often the *optimal* policy for a signal-free agent — Leap's "degenerate" result may be the correct answer to the question being asked.

7. **Multiple testing.** Feature sets × architectures × reward penalties × online-learning schedules × regime detectors constitute hundreds of trials. Harvey & Liu (2015) and Bailey & López de Prado (2014) show the expected maximum Sharpe under the null grows with √(2 ln N); with N ≈ 100 trials on a few years of data an in-sample Sharpe near 1 is expected by pure chance.

### What a realistic target looks like
| Design | Evidence | Realistic net Sharpe (after retail costs) |
|---|---|---|
| Single pair, M15/H1, technical features, any learner | Neely & Weller 2003; Curcio 1997; Kozhan & Salmon 2010; Wood 2022 FX rows; cost arithmetic above | **≈0 or negative**; occasional lucky year |
| Single pair, daily bars, TSMOM/carry with vol targeting | MOP 2012 per-instrument results; Wood 2022 FX 0.28 gross; Menkhoff 2012 | **0–0.3**, multi-year drawdowns; unattractive alone |
| 20–50 MT5 instruments (FX majors/crosses, metals, indices, energies), daily, TSMOM + carry + vol targeting | MOP 2012; HOP 2012/17 (1.0 net of fees, 0.4 conservative); ZZR 2020 (1.3 gross at 25 bp on 50 futures) | **0.4–0.8 net** is a defensible expectation; >1 would be exceptional |
| Same universe + learned sizing (DMN / Momentum Transformer) | Lim 2019 (~2× TSMOM gross to 2–3 bps); Wood 2022 (portfolio 1.0–1.2 at 1 bp) | Adds perhaps +0.2–0.5 gross **only if** CFD costs ≲1–2 bps and turnover is regularised; retail CFD swaps/spreads often exceed this |

The honest single-pair goal is: *demonstrate, with DSR/SPA at the t > 3 level, that the system does not lose money after costs and is not distinguishable from a vol-targeted TSMOM baseline* — then expand the universe.

---

## 4. Concrete redesign recommendations (each tied to its evidence)

1. **Change the objective from "predict next bar" to "run a risk-targeted portfolio."** Position_i,t = clip(signal_i,t, −1, 1) × σ_target / σ̂_i,t, aggregated to a portfolio vol target (10% p.a.), rebalanced daily. — MOP (2012); HOP (2012/17); ZZR (2020, formula in §3.1 of paper); Lim et al. (2019).

2. **Expand the universe to everything liquid on the MT5 account** (all G10 pairs and liquid crosses, XAU/XAG, major index CFDs, oil), and run *both* time-series (sign of 1–12-month return) and cross-sectional (rank) signals; add **carry** (swap/interest differential) as a signal and as a feature. — MOP; Menkhoff et al. (2012); Rossi (2013) on interest-differential predictors. Filter out instruments whose round-trip spread exceeds ~10% of daily E|r|.

3. **Labels: triple-barrier + meta-labelling if a supervised classifier is kept.** Primary signal = TSMOM/MACD/carry rule; secondary GBM learns P(primary signal is right) from vol, regime, spread, session-time features; bet size ∝ calibrated probability. — López de Prado (2018) chs. 3, 10; López-Herrera et al. (2025) on directional/profit-oriented losses beating black-box models.

4. **Model ladder with mandatory baselines.** Persistence/zero → DLinear/NLinear → ridge/logistic → LightGBM/XGBoost → (optional) small LSTM/TFT. Promote a model only if it beats the previous rung on **pinball/MASE relative to persistence** *and* on net PnL under Diebold-Mariano/SPA. — Zeng et al. (2023); GKX (2020); M4/M5; Fischer & Krauss (2018); Krauss et al. (2017).

5. **Drop PPO as the decision layer.** Replace with (a) the deterministic vol-targeting rule above, or (b) a single network trained end-to-end on a Sharpe/utility loss with turnover penalty that outputs position directly (DMN). If RL is retained for research: vol-scaled *additive* reward with explicit bps cost term, daily bars, ≥20 instruments, 5+ seeds ensembled, agents screened by PBO before any evaluation, and A2C/DQN rather than PPO given ZZR's results. — Lim et al. (2019); Wood et al. (2022); ZZR (2020); Gort et al. (2022); Mohammadshafie et al. (2024).

6. **Validation protocol (non-negotiable).** Purged, embargoed walk-forward with ≥5 folds; CPCV to generate ≥50 backtest paths; record every trial; report **Deflated Sharpe** (p-value vs. N trials), **PBO** (target < 0.2), **Hansen SPA** vs. {no-trade, buy-and-hold, TSMOM}; net-of-cost results by sub-period; require the HLZ t > 3 hurdle before paper trading. — López de Prado (2018) chs. 7, 11–12; Bailey et al. (2014, 2017); Harvey & Liu (2015); HLZ (2016); White (2000); Hansen (2005).

7. **Transaction-cost model in every layer.** Charge the live spread distribution (time-of-day dependent, wider at news), commission per lot, swap for overnight holds, and 0.2–0.5 pip slippage; publish Sharpe-vs-cost curves from 0 to 2× current cost as Wood et al. do; restrict trading to London/NY overlap as Neely & Weller (2003) did. — Wood et al. (2022) Exhibit 10; Lim et al. (2019); Neely & Weller (2003).

8. **Regime handling: use it for risk, not for alpha.** Fit a 2-state Hamilton (1989) Markov-switching model on daily realised volatility (or use realised-vol quantiles); scale exposure down in the high-vol state; optionally feed change-point scores as features (Wood et al. 2022, +17% Sharpe). Stop online re-training of the forecaster; retrain on a fixed schedule with purging. — Hamilton (1989); Ang & Timmermann (2012); Wood et al. (2022).

9. **Feature hygiene.** Replace 50–100 indicators with ~10–15: normalised returns at 1d/1w/1m/3m/6m/12m, MACD at 3 speeds, realised vol at 2 speeds, carry, spread, session dummies; fractionally differentiate any price-level feature; prune by MDA importance on purged CV. — AFML chs. 5, 8; ZZR/Lim/Wood feature sets.

10. **Reporting.** For every run: net Sharpe, DSR p-value, PBO, max drawdown, turnover, cost as % of gross PnL, and the same for each baseline; a result is "positive" only if it beats TSMOM on SPA at 5%. — Harvey & Liu (2015); Bailey & López de Prado (2014).

---

## 5. Verification notes and limitations of this review

- Verified by fetching abstract pages and/or extracting PDF text: Rossi 2013 (AEA/CREI), Meese-Rogoff 1983 (Harvard PDF listing), Zeng 2023 (arXiv), Zhang-Zohren-Roberts 2020 (arXiv PDF text; Table 2 numbers quoted), Gu-Kelly-Xiu 2020 (Chicago Booth PDF text; R² and Sharpe quoted), Wood et al. 2022 (arXiv PDF text; Exhibits 3 and 10 quoted), Hurst-Ooi-Pedersen 2012 white paper (PDF text; Exhibit 1 quoted), Neely & Weller 2011 (St. Louis Fed PDF text; quotations exact), Guyard & Deriaz 2024 (arXiv abstract), López-Herrera et al. 2025 (Semantic Scholar record), Gort et al. 2022 (arXiv abstract), Mohammadshafie et al. 2024 (arXiv HTML), Millea 2021 (Crossref record), FinRL 2021 (arXiv listing).
- Verified by search-result snippets/citations only [not independently fetched]: Lim et al. 2019 DMN figures ("88 contracts", "2–3 bps"), Lim et al. 2021 TFT, Menkhoff et al. 2012 ("up to 10% p.a.", cost sensitivity), Moskowitz-Ooi-Pedersen 2012 (58 instruments), Fischer & Krauss 2018 (0.46%/day, Sharpe 5.8 pre-cost, post-2010 decay), Krauss et al. 2017 (0.45%/day), Sezer et al. 2020, Bailey et al. PBO/DSR (bibliographic), Harvey & Liu 2015 / HLZ 2016, White 2000, Hansen 2005, Hamilton 1989, Ang & Timmermann 2012, Neely-Weller-Ulrich 2009, Curcio et al. 1997 (via Neely & Weller 2011 bibliography), Makridakis M4/M5, Chaboud et al. 2009 and Kozhan & Salmon 2010 (via Neely & Weller 2011). The Hurst-Ooi-Pedersen figures quoted are from the **2012** AQR white-paper version (1903–2012); the 2017 JPM version extends to 1880–2016 and reports somewhat different (still ≈0.7–1.0 gross-of-fee) numbers that I did not extract.
- Retail spread figures come from broker-comparison websites (practitioner sources, 2026), not peer-reviewed work; the break-even accuracy table is my own arithmetic under a 7% annualised-vol assumption — recompute with Leap's realised σ per bar and live spread logs.
- I did not find a peer-reviewed paper that specifically documents "PPO converges to always-hold in FX"; the closest verified evidence is Mohammadshafie et al. (2024) on holding behaviour across DRL algorithms, ZZR's reasoning on reward scaling, and the survey literature on reward-design sensitivity.
- No files in /home/user/Leap were read or modified; this review is based on the project description supplied.

Local artefacts: extracted paper texts in /tmp/claude-0/-home-user-Leap/c1215b41-bae4-5a3d-a328-ca81e80bfaf3/scratchpad/{zzr_a,hop,neely,gkx,momtf}.txt
