"""
Drovalix AI Score Engine
File: ai_engine/score_stock.py

A highly detailed, extensible stock scoring system using Yahoo Finance data (via yfinance).
This version includes comprehensive metric coverage: financial, growth, value, liquidity, technical, sentiment, longevity, and risk.
Features:
- 35+ modular scoring metrics (financial, value, technical, sentiment, risk, longevity, quality, etc).
- Peerless CLI and batch scoring with JSON/CSV/Markdown/PrettyTable export.
- Logging, error handling, debug mode.
- Rich documentation and ready for unit/integration testing.
- Easily extensible for custom metrics, sector/peer comparison, and more.

Author: Drovalix AI Team
"""

import yfinance as yf
import sys
import json
import csv
import argparse
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

try:
    from prettytable import PrettyTable
except ImportError:
    PrettyTable = None

# =========================
# Logging Configuration
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DrovalixScoreEngine")

# =========================
# Metric Base Class
# =========================
class StockMetric:
    name: str = "GenericMetric"
    max_points: int = 0
    description: str = ""
    weight: float = 1.0

    def score(self, info: Dict[str, Any]) -> Tuple[int, List[str]]:
        raise NotImplementedError("score() must be implemented in subclasses.")

    def explain(self) -> str:
        return self.description

    def required_keys(self) -> List[str]:
        return []

# =========================
# Metric Implementations (Extensive Set)
# =========================

# Financial Quality
class ROEMetric(StockMetric):
    name = "Return on Equity"
    max_points = 20
    weight = 1.2
    description = "Measures profitability relative to shareholder equity."

    def score(self, info):
        points = 0
        reasons = []
        roe = info.get('returnOnEquity')
        if roe is not None:
            if roe > 0.20:
                points += 20
                reasons.append(f"Outstanding ROE of {roe:.2f} (>20%)")
            elif roe > 0.15:
                points += 15
                reasons.append(f"Strong ROE of {roe:.2f} (>15%)")
            elif roe > 0.10:
                points += 10
                reasons.append(f"Moderate ROE of {roe:.2f} (>10%)")
            else:
                reasons.append(f"Low ROE of {roe:.2f} (≤10%)")
        else:
            reasons.append("ROE data not available")
        return points, reasons

class ReturnOnAssetsMetric(StockMetric):
    name = "Return on Assets"
    max_points = 3
    description = "Indicates efficient asset use (ROA > 10% is excellent)."
    def score(self, info):
        points = 0
        reasons = []
        roa = info.get('returnOnAssets')
        if roa is not None:
            if roa > 0.10:
                points += 3
                reasons.append(f"Excellent ROA: {roa*100:.2f}% (>10%)")
            elif roa > 0.05:
                points += 2
                reasons.append(f"Good ROA: {roa*100:.2f}% (>5%)")
            elif roa > 0:
                points += 1
                reasons.append(f"Positive ROA: {roa*100:.2f}%")
            else:
                reasons.append(f"Negative or zero ROA: {roa*100:.2f}%")
        else:
            reasons.append("ROA data not available")
        return points, reasons

class ROICMetric(StockMetric):
    name = "Return on Invested Capital"
    max_points = 4
    description = "High ROIC (>10%) means efficient capital allocation."
    def score(self, info):
        points = 0
        reasons = []
        roic = info.get('returnOnInvestedCapital')
        if roic is not None:
            if roic > 0.15:
                points += 4
                reasons.append(f"Outstanding ROIC: {roic*100:.2f}% (>15%)")
            elif roic > 0.10:
                points += 2
                reasons.append(f"Good ROIC: {roic*100:.2f}% (>10%)")
            elif roic > 0:
                points += 1
                reasons.append(f"Positive ROIC: {roic*100:.2f}%")
            else:
                reasons.append(f"Negative or zero ROIC: {roic*100:.2f}%")
        else:
            reasons.append("ROIC data not available")
        return points, reasons

# Liquidity/Solvency
class DebtToEquityMetric(StockMetric):
    name = "Debt to Equity"
    max_points = 15
    description = "Assesses leverage: lower is safer."
    def score(self, info):
        points = 0
        reasons = []
        dte = info.get('debtToEquity')
        if dte is not None:
            if dte < 0.5:
                points += 15
                reasons.append(f"Excellent D/E: {dte:.2f} (<0.5)")
            elif dte < 1:
                points += 10
                reasons.append(f"Healthy D/E: {dte:.2f} (<1)")
            elif dte < 2:
                points += 5
                reasons.append(f"Acceptable D/E: {dte:.2f} (<2)")
            else:
                reasons.append(f"High D/E: {dte:.2f} (≥2)")
        else:
            reasons.append("Debt/Equity data not available")
        return points, reasons

class CurrentRatioMetric(StockMetric):
    name = "Current Ratio"
    max_points = 7
    description = "Short-term liquidity (current assets / current liabilities)."
    def score(self, info):
        points = 0
        reasons = []
        cr = info.get('currentRatio')
        if cr is not None:
            if cr > 2:
                points += 7
                reasons.append(f"Very strong current ratio {cr:.2f} (>2)")
            elif cr > 1.5:
                points += 5
                reasons.append(f"Good current ratio {cr:.2f} (>1.5)")
            elif cr > 1:
                points += 3
                reasons.append(f"Acceptable current ratio {cr:.2f} (>1)")
            else:
                reasons.append(f"Low current ratio {cr:.2f} (≤1)")
        else:
            reasons.append("Current ratio data not available")
        return points, reasons

class QuickRatioMetric(StockMetric):
    name = "Quick Ratio"
    max_points = 6
    description = "Liquidity (quick assets / current liabilities)."
    def score(self, info):
        points = 0
        reasons = []
        qr = info.get('quickRatio')
        if qr is not None:
            if qr > 1.5:
                points += 6
                reasons.append(f"Excellent quick ratio {qr:.2f} (>1.5)")
            elif qr > 1.0:
                points += 4
                reasons.append(f"Good quick ratio {qr:.2f} (>1.0)")
            elif qr > 0.7:
                points += 2
                reasons.append(f"Acceptable quick ratio {qr:.2f} (>0.7)")
            else:
                reasons.append(f"Weak quick ratio {qr:.2f} (≤0.7)")
        else:
            reasons.append("Quick ratio data not available")
        return points, reasons

class InterestCoverageMetric(StockMetric):
    name = "Interest Coverage"
    max_points = 3
    description = "EBIT/Interest > 4 is safe (debt payments)."
    def score(self, info):
        points = 0
        reasons = []
        ebit = info.get('ebit')
        interest_exp = info.get('interestExpense')
        if ebit is not None and interest_exp is not None and interest_exp != 0:
            coverage = ebit / abs(interest_exp)
            if coverage > 8:
                points += 3
                reasons.append(f"Excellent Interest Coverage: {coverage:.1f}x (>8x)")
            elif coverage > 4:
                points += 2
                reasons.append(f"Good Interest Coverage: {coverage:.1f}x (>4x)")
            elif coverage > 2:
                points += 1
                reasons.append(f"Acceptable Interest Coverage: {coverage:.1f}x (>2x)")
            else:
                reasons.append(f"Low Interest Coverage: {coverage:.1f}x (≤2x)")
        else:
            reasons.append("Insufficient data for interest coverage")
        return points, reasons

# Profitability
class ProfitMarginMetric(StockMetric):
    name = "Profit Margin"
    max_points = 12
    description = "Profit as a % of revenue."
    def score(self, info):
        points = 0
        reasons = []
        pm = info.get('profitMargins')
        if pm is not None:
            pm_percent = pm * 100
            if pm_percent > 20:
                points += 12
                reasons.append(f"Excellent profit margin {pm_percent:.2f}% (>20%)")
            elif pm_percent > 10:
                points += 8
                reasons.append(f"Good profit margin {pm_percent:.2f}% (>10%)")
            elif pm_percent > 5:
                points += 4
                reasons.append(f"Thin profit margin {pm_percent:.2f}% (>5%)")
            else:
                reasons.append(f"Very thin profit margin {pm_percent:.2f}% (≤5%)")
        else:
            reasons.append("Profit margin data not available")
        return points, reasons

class OperatingMarginMetric(StockMetric):
    name = "Operating Margin"
    max_points = 10
    description = "Operating profit as % of revenue."
    def score(self, info):
        points = 0
        reasons = []
        opm = info.get('operatingMargins')
        if opm is not None:
            opm_percent = opm * 100
            if opm_percent > 20:
                points += 10
                reasons.append(f"Strong operating margin {opm_percent:.2f}% (>20%)")
            elif opm_percent > 10:
                points += 6
                reasons.append(f"Good operating margin {opm_percent:.2f}% (>10%)")
            else:
                reasons.append(f"Weak operating margin {opm_percent:.2f}% (≤10%)")
        else:
            reasons.append("Operating margin data not available")
        return points, reasons

class FreeCashFlowMetric(StockMetric):
    name = "Free Cash Flow"
    max_points = 10
    description = "Positive FCF is rewarded."
    def score(self, info):
        points = 0
        reasons = []
        fcf = info.get('freeCashflow')
        if fcf is not None:
            if fcf > 0:
                points += 10
                reasons.append("Positive Free Cash Flow")
            else:
                reasons.append("Negative Free Cash Flow")
        else:
            reasons.append("Free Cash Flow data not available")
        return points, reasons

class PriceToFreeCashFlowMetric(StockMetric):
    name = "P/FCF Ratio"
    max_points = 3
    description = "P/FCF < 15 is attractive."
    def score(self, info):
        points = 0
        reasons = []
        mcap = info.get('marketCap')
        fcf = info.get('freeCashflow')
        if mcap and fcf and fcf > 0:
            pfcf = mcap / fcf
            if pfcf < 10:
                points += 3
                reasons.append(f"Very attractive P/FCF: {pfcf:.2f} (<10)")
            elif pfcf < 15:
                points += 2
                reasons.append(f"Attractive P/FCF: {pfcf:.2f} (<15)")
            else:
                reasons.append(f"High P/FCF: {pfcf:.2f} (≥15)")
        else:
            reasons.append("Not enough data for P/FCF")
        return points, reasons

# Growth
class RevenueGrowthMetric(StockMetric):
    name = "Revenue Growth"
    max_points = 10
    description = "Year-over-year revenue growth."
    def score(self, info):
        points = 0
        reasons = []
        rev_growth = info.get('revenueGrowth')
        if rev_growth is not None:
            pct = rev_growth * 100
            if pct > 20:
                points += 10
                reasons.append(f"Exceptional revenue growth {pct:.2f}% (>20%)")
            elif pct > 10:
                points += 7
                reasons.append(f"Strong revenue growth {pct:.2f}% (>10%)")
            elif pct > 0:
                points += 4
                reasons.append(f"Positive revenue growth {pct:.2f}%")
            else:
                reasons.append(f"Negative revenue growth {pct:.2f}%")
        else:
            reasons.append("Revenue growth data not available")
        return points, reasons

class FiveYearRevenueGrowthMetric(StockMetric):
    name = "5y Revenue CAGR"
    max_points = 5
    description = "Sustained 5-year revenue CAGR."
    def score(self, info):
        points = 0
        reasons = []
        rev5y = info.get('fiveYearAvgRevenueGrowth')
        if rev5y is not None:
            rev5y_pct = rev5y * 100
            if rev5y_pct > 15:
                points += 5
                reasons.append(f"Outstanding 5-year revenue CAGR: {rev5y_pct:.2f}% (>15%)")
            elif rev5y_pct > 7:
                points += 3
                reasons.append(f"Good 5-year revenue CAGR: {rev5y_pct:.2f}% (>7%)")
            elif rev5y_pct > 0:
                points += 1
                reasons.append(f"Positive 5-year revenue CAGR: {rev5y_pct:.2f}%")
            else:
                reasons.append(f"Negative 5-year revenue CAGR: {rev5y_pct:.2f}%")
        else:
            reasons.append("5-year revenue growth data not available")
        return points, reasons

class EPSGrowthMetric(StockMetric):
    name = "EPS Growth"
    max_points = 8
    description = "YOY earnings per share growth."
    def score(self, info):
        points = 0
        reasons = []
        eps_growth = info.get('earningsQuarterlyGrowth')
        if eps_growth is not None:
            pct = eps_growth * 100
            if pct > 15:
                points += 8
                reasons.append(f"Excellent EPS growth {pct:.2f}% (>15%)")
            elif pct > 5:
                points += 4
                reasons.append(f"Positive EPS growth {pct:.2f}%")
            else:
                reasons.append(f"Minimal EPS growth {pct:.2f}% (≤5%)")
        else:
            reasons.append("EPS growth data not available")
        return points, reasons

class DividendGrowthMetric(StockMetric):
    name = "Dividend Growth"
    max_points = 4
    description = "Rewards 3y+ consecutive dividend growth."
    def score(self, info):
        points = 0
        reasons = []
        dg = info.get('dividendGrowth')
        if dg is not None:
            if dg >= 5:
                points += 4
                reasons.append(f"{dg} years of dividend growth (≥5y)")
            elif dg >= 3:
                points += 2
                reasons.append(f"{dg} years of dividend growth (≥3y)")
            else:
                reasons.append(f"Dividend growth streak: {dg} years (<3y)")
        else:
            reasons.append("Dividend growth data not available")
        return points, reasons

# Valuation
class PERatioMetric(StockMetric):
    name = "P/E Ratio"
    max_points = 5
    description = "Low P/E (<15) is rewarded."
    def score(self, info):
        points = 0
        reasons = []
        pe = info.get('trailingPE')
        if pe is not None and pe > 0:
            if pe < 15:
                points += 5
                reasons.append(f"Low P/E: {pe:.2f} (<15)")
            elif pe < 25:
                points += 3
                reasons.append(f"Reasonable P/E: {pe:.2f} (<25)")
            else:
                reasons.append(f"High P/E: {pe:.2f} (≥25)")
        else:
            reasons.append("P/E ratio data not available or negative")
        return points, reasons

class PEGMetric(StockMetric):
    name = "PEG Ratio"
    max_points = 3
    description = "PEG < 1 is undervalued for growth."
    def score(self, info):
        points = 0
        reasons = []
        peg = info.get('pegRatio')
        if peg is not None and peg > 0:
            if peg < 1:
                points += 3
                reasons.append(f"Low PEG: {peg:.2f} (<1, undervalued)")
            elif peg < 2:
                points += 1
                reasons.append(f"Reasonable PEG: {peg:.2f} (<2)")
            else:
                reasons.append(f"High PEG: {peg:.2f} (≥2)")
        else:
            reasons.append("PEG ratio data not available or N/A")
        return points, reasons

class PBMetric(StockMetric):
    name = "P/B Ratio"
    max_points = 5
    description = "Low P/B (<2) is rewarded."
    def score(self, info):
        points = 0
        reasons = []
        pb = info.get('priceToBook')
        if pb is not None and pb > 0:
            if pb < 2:
                points += 5
                reasons.append(f"Low P/B: {pb:.2f} (<2)")
            elif pb < 4:
                points += 2
                reasons.append(f"Reasonable P/B: {pb:.2f} (<4)")
            else:
                reasons.append(f"High P/B: {pb:.2f} (≥4)")
        else:
            reasons.append("P/B ratio data not available or negative")
        return points, reasons

class PriceToSalesMetric(StockMetric):
    name = "P/S Ratio"
    max_points = 4
    description = "P/S < 2 is best."
    def score(self, info):
        points = 0
        reasons = []
        ps = info.get('priceToSalesTrailing12Months')
        if ps is not None and ps > 0:
            if ps < 2:
                points += 4
                reasons.append(f"Low P/S: {ps:.2f} (<2)")
            elif ps < 4:
                points += 2
                reasons.append(f"Reasonable P/S: {ps:.2f} (<4)")
            else:
                reasons.append(f"High P/S: {ps:.2f} (≥4)")
        else:
            reasons.append("P/S ratio data not available or N/A")
        return points, reasons

class GrahamNumberMetric(StockMetric):
    name = "Graham Number"
    max_points = 4
    description = "Rewards stocks trading below Graham Number (undervalued)."
    def score(self, info):
        points = 0
        reasons = []
        eps = info.get('trailingEps')
        bvps = info.get('bookValue')
        price = info.get('currentPrice')
        if eps and bvps and price:
            graham = (22.5 * eps * bvps) ** 0.5
            if price < graham:
                points += 4
                reasons.append("Trading below Graham Number (undervalued)")
            else:
                reasons.append("Trading above Graham Number (not undervalued)")
        else:
            reasons.append("Not enough data for Graham Number")
        return points, reasons

class PriceToFreeCashFlowMetric(StockMetric):
    name = "P/FCF Ratio"
    max_points = 3
    description = "P/FCF < 15 rewarded."
    def score(self, info):
        points = 0
        reasons = []
        mcap = info.get('marketCap')
        fcf = info.get('freeCashflow')
        if mcap and fcf and fcf > 0:
            pfcf = mcap / fcf
            if pfcf < 10:
                points += 3
                reasons.append(f"Very attractive P/FCF: {pfcf:.2f} (<10)")
            elif pfcf < 15:
                points += 2
                reasons.append(f"Attractive P/FCF: {pfcf:.2f} (<15)")
            else:
                reasons.append(f"High P/FCF: {pfcf:.2f} (≥15)")
        else:
            reasons.append("Not enough data for P/FCF")
        return points, reasons

# Dividend / Payout
class DividendMetric(StockMetric):
    name = "Dividend Yield"
    max_points = 5
    description = "Rewards decent dividend yield."
    def score(self, info):
        points = 0
        reasons = []
        div_yield = info.get('dividendYield')
        if div_yield is not None:
            pct = div_yield * 100
            if pct > 3:
                points += 5
                reasons.append(f"Attractive dividend yield {pct:.2f}% (>3%)")
            elif pct > 1:
                points += 2
                reasons.append(f"Modest dividend yield {pct:.2f}%")
            else:
                reasons.append(f"Low dividend yield {pct:.2f}% (≤1%)")
        else:
            reasons.append("Dividend yield data not available")
        return points, reasons

class PayoutRatioMetric(StockMetric):
    name = "Payout Ratio"
    max_points = 3
    description = "Payout ratio <60% is sustainable."
    def score(self, info):
        points = 0
        reasons = []
        payout = info.get('payoutRatio')
        if payout is not None and payout > 0:
            payout_pct = payout * 100
            if payout < 0.4:
                points += 3
                reasons.append(f"Conservative payout ratio: {payout_pct:.1f}% (<40%)")
            elif payout < 0.6:
                points += 2
                reasons.append(f"Manageable payout ratio: {payout_pct:.1f}% (<60%)")
            else:
                reasons.append(f"High payout ratio: {payout_pct:.1f}% (≥60%)")
        else:
            reasons.append("Payout ratio data not available or N/A")
        return points, reasons

# Sentiment, Insider, Institutional
class ShortFloatMetric(StockMetric):
    name = "Short Interest %"
    max_points = 4
    description = "Penalizes high short interest (risk/negative sentiment)."
    def score(self, info):
        points = 0
        reasons = []
        short_percent = info.get('shortPercentOfFloat')
        if short_percent is not None:
            if short_percent < 0.02:
                points += 4
                reasons.append(f"Very low short interest: {short_percent*100:.2f}% (<2%)")
            elif short_percent < 0.05:
                points += 2
                reasons.append(f"Low short interest: {short_percent*100:.2f}% (<5%)")
            elif short_percent < 0.10:
                points += 1
                reasons.append(f"Moderate short interest: {short_percent*100:.2f}% (<10%)")
            else:
                reasons.append(f"High short interest: {short_percent*100:.2f}% (≥10%)")
        else:
            reasons.append("Short interest data not available")
        return points, reasons

class AnalystRecommendationMetric(StockMetric):
    name = "Analyst Recommendation"
    max_points = 5
    description = "Strong buy/buy consensus rewarded."
    def score(self, info):
        points = 0
        reasons = []
        reco = info.get('recommendationKey')
        if reco is not None:
            if reco == "strong_buy":
                points += 5
                reasons.append("Analyst consensus: Strong Buy")
            elif reco == "buy":
                points += 3
                reasons.append("Analyst consensus: Buy")
            elif reco == "hold":
                points += 1
                reasons.append("Analyst consensus: Hold")
            else:
                reasons.append(f"Analyst consensus: {reco.replace('_', ' ').title()}")
        else:
            reasons.append("Analyst recommendation not available")
        return points, reasons

class InsiderOwnershipMetric(StockMetric):
    name = "Insider Ownership"
    max_points = 5
    description = "Rewards substantial insider ownership."
    def score(self, info):
        points = 0
        reasons = []
        insider_percent = info.get('heldPercentInsiders')
        if insider_percent is not None:
            if insider_percent > 0.1:
                points += 5
                reasons.append(f"High insider ownership: {insider_percent * 100:.2f}% (>10%)")
            elif insider_percent > 0.03:
                points += 3
                reasons.append(f"Moderate insider ownership: {insider_percent * 100:.2f}% (>3%)")
            else:
                reasons.append(f"Low insider ownership: {insider_percent * 100:.2f}% (≤3%)")
        else:
            reasons.append("Insider ownership data not available")
        return points, reasons

class InstitutionalOwnershipMetric(StockMetric):
    name = "Institutional Ownership"
    max_points = 5
    description = "Rewards strong institutional backing."
    def score(self, info):
        points = 0
        reasons = []
        ii_percent = info.get('heldPercentInstitutions')
        if ii_percent is not None:
            if ii_percent > 0.7:
                points += 5
                reasons.append(f"High institutional ownership: {ii_percent * 100:.2f}% (>70%)")
            elif ii_percent > 0.4:
                points += 3
                reasons.append(f"Moderate institutional ownership: {ii_percent * 100:.2f}% (>40%)")
            else:
                reasons.append(f"Low institutional ownership: {ii_percent * 100:.2f}% (≤40%)")
        else:
            reasons.append("Institutional ownership data not available")
        return points, reasons

# ESG and Risk
class ESGScoreMetric(StockMetric):
    name = "ESG Score"
    max_points = 5
    description = "Environmental/Social/Governance risk score."
    def score(self, info):
        points = 0
        reasons = []
        esg = info.get('esgScores')
        if esg and isinstance(esg, dict):
            combined_score = esg.get('totalEsg')
            if combined_score is not None:
                if combined_score < 25:
                    points += 5
                    reasons.append(f"Excellent ESG risk score: {combined_score:.1f} (<25)")
                elif combined_score < 40:
                    points += 3
                    reasons.append(f"Good ESG risk score: {combined_score:.1f} (<40)")
                else:
                    reasons.append(f"High ESG risk score: {combined_score:.1f} (≥40, higher is worse)")
            else:
                reasons.append("Total ESG score not available")
        else:
            reasons.append("ESG data not available")
        return points, reasons

class AltmanZScoreMetric(StockMetric):
    name = "Altman Z-Score"
    max_points = 6
    description = "Bankruptcy risk (higher is safer)."
    def score(self, info):
        points = 0
        reasons = []
        try:
            wc_ta = info.get('totalCurrentAssets', 0) - info.get('totalCurrentLiabilities', 0)
            total_assets = info.get('totalAssets', 0)
            retained_earnings = info.get('retainedEarnings', 0)
            ebit = info.get('ebit', 0)
            market_cap = info.get('marketCap', 0)
            total_liabilities = info.get('totalLiab', 0)
            sales = info.get('totalRevenue', 0)
            if all(x > 0 for x in [total_assets, total_liabilities, sales, market_cap]):
                z = (
                    1.2 * (wc_ta / total_assets) +
                    1.4 * (retained_earnings / total_assets) +
                    3.3 * (ebit / total_assets) +
                    0.6 * (market_cap / total_liabilities) +
                    1.0 * (sales / total_assets)
                )
                if z > 3.0:
                    points += 6
                    reasons.append(f"Very safe Altman Z-Score: {z:.2f} (>3.0)")
                elif z > 2.5:
                    points += 4
                    reasons.append(f"Safe Altman Z-Score: {z:.2f} (>2.5)")
                elif z > 1.8:
                    points += 2
                    reasons.append(f"Warning Altman Z-Score: {z:.2f} (>1.8)")
                else:
                    reasons.append(f"Distress Altman Z-Score: {z:.2f} (≤1.8)")
            else:
                reasons.append("Insufficient data for Altman Z-Score")
        except Exception:
            reasons.append("Could not compute Altman Z-Score")
        return points, reasons

# Size, Liquidity, Volatility
class MarketCapMetric(StockMetric):
    name = "Market Capitalization"
    max_points = 5
    description = "Rewards large and stable companies."
    def score(self, info):
        points = 0
        reasons = []
        mcap = info.get('marketCap')
        if mcap is not None:
            if mcap > 1e11:
                points += 5
                reasons.append(f"Very large market cap: ${mcap/1e9:.1f}B")
            elif mcap > 1e10:
                points += 3
                reasons.append(f"Large market cap: ${mcap/1e9:.1f}B")
            elif mcap > 1e9:
                points += 1
                reasons.append(f"Mid cap: ${mcap/1e9:.1f}B")
            else:
                reasons.append(f"Small cap: ${mcap/1e6:.1f}M")
        else:
            reasons.append("Market cap data not available")
        return points, reasons

class AvgVolumeMetric(StockMetric):
    name = "Avg Volume (Liquidity)"
    max_points = 4
    description = "Rewards daily trading liquidity."
    def score(self, info):
        points = 0
        reasons = []
        avgvol = info.get('averageDailyVolume10Day')
        if avgvol is not None:
            if avgvol >= 1_000_000:
                points += 4
                reasons.append(f"Excellent liquidity: {avgvol:,}/day")
            elif avgvol >= 300_000:
                points += 2
                reasons.append(f"Acceptable liquidity: {avgvol:,}/day")
            else:
                reasons.append(f"Low liquidity: {avgvol:,}/day (<300k)")
        else:
            reasons.append("Average volume data not available")
        return points, reasons

class BetaMetric(StockMetric):
    name = "Beta (Volatility)"
    max_points = 4
    description = "Rewards lower-than-market volatility."
    def score(self, info):
        points = 0
        reasons = []
        beta = info.get('beta')
        if beta is not None:
            if 0 < beta < 1:
                points += 4
                reasons.append(f"Lower-than-market volatility (beta={beta:.2f})")
            elif 1 <= beta < 1.3:
                points += 2
                reasons.append(f"Market-level volatility (beta={beta:.2f})")
            else:
                reasons.append(f"High volatility (beta={beta:.2f})")
        else:
            reasons.append("Beta data not available")
        return points, reasons

# Technicals & Longevity
class PriceMomentumMetric(StockMetric):
    name = "1y Price Momentum"
    max_points = 3
    description = "Rewards positive 12-month price performance."
    def score(self, info):
        points = 0
        reasons = []
        symbol = info.get('symbol')
        if not symbol:
            reasons.append("Ticker symbol not available for price momentum metric")
            return points, reasons
        try:
            hist = yf.Ticker(symbol).history(period="1y")
            if not hist.empty:
                price_change = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]
                pct = price_change * 100
                if pct > 30:
                    points += 3
                    reasons.append(f"Strong 1y price momentum: {pct:.1f}% (>30%)")
                elif pct > 10:
                    points += 2
                    reasons.append(f"Good 1y price momentum: {pct:.1f}% (>10%)")
                elif pct > 0:
                    points += 1
                    reasons.append(f"Positive 1y price momentum: {pct:.1f}%")
                else:
                    reasons.append(f"Negative 1y price momentum: {pct:.1f}%")
            else:
                reasons.append("No 1y price history available")
        except Exception:
            reasons.append("Failed to compute 1y price momentum")
        return points, reasons

class CompanyAgeMetric(StockMetric):
    name = "Company Longevity"
    max_points = 3
    description = "Rewards older, established companies."
    def score(self, info):
        points = 0
        reasons = []
        ipo_year = info.get('ipoYear')
        if ipo_year is None:
            start_date = info.get('startDate')
            if start_date:
                try:
                    ipo_year = int(str(start_date)[:4])
                except Exception:
                    ipo_year = None
        if ipo_year:
            current_year = datetime.utcnow().year
            age = current_year - int(ipo_year)
            if age > 30:
                points += 3
                reasons.append(f"Mature company (age={age}y, IPO {ipo_year})")
            elif age > 10:
                points += 2
                reasons.append(f"Established company (age={age}y, IPO {ipo_year})")
            elif age > 3:
                points += 1
                reasons.append(f"Newer public company (age={age}y, IPO {ipo_year})")
            else:
                reasons.append(f"Very recent IPO (age={age}y, IPO {ipo_year})")
        else:
            reasons.append("IPO year/start date not available")
        return points, reasons

# =========================
# DrovalixScorer
# =========================

class DrovalixScorer:
    def __init__(self, metrics: Optional[List[StockMetric]] = None):
        if metrics is None:
            metrics = [
                ROEMetric(),
                ReturnOnAssetsMetric(),
                ROICMetric(),
                DebtToEquityMetric(),
                CurrentRatioMetric(),
                QuickRatioMetric(),
                InterestCoverageMetric(),
                ProfitMarginMetric(),
                OperatingMarginMetric(),
                FreeCashFlowMetric(),
                PriceToFreeCashFlowMetric(),
                RevenueGrowthMetric(),
                FiveYearRevenueGrowthMetric(),
                EPSGrowthMetric(),
                DividendGrowthMetric(),
                PERatioMetric(),
                PEGMetric(),
                PBMetric(),
                PriceToSalesMetric(),
                GrahamNumberMetric(),
                DividendMetric(),
                PayoutRatioMetric(),
                ShortFloatMetric(),
                AnalystRecommendationMetric(),
                InsiderOwnershipMetric(),
                InstitutionalOwnershipMetric(),
                ESGScoreMetric(),
                AltmanZScoreMetric(),
                MarketCapMetric(),
                AvgVolumeMetric(),
                BetaMetric(),
                PriceMomentumMetric(),
                CompanyAgeMetric(),
            ]
        self.metrics = metrics
        self.max_score = sum(metric.max_points for metric in self.metrics)

    def get_rating(self, score: int) -> str:
        pct = (score / self.max_score) * 100 if self.max_score else 0
        if pct >= 85:
            return "Excellent"
        elif pct >= 70:
            return "Very Good"
        elif pct >= 55:
            return "Good"
        elif pct >= 40:
            return "Average"
        else:
            return "Weak"

    def score_stock(self, ticker: str, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        logger.info(f"Scoring ticker: {ticker}")
        try:
            stock = yf.Ticker(ticker)
            if info is None:
                info = stock.info
            info['symbol'] = ticker  # for technical and price metrics
            total_score = 0
            reasons: List[str] = []
            metric_breakdown: List[Dict[str, Any]] = []
            for metric in self.metrics:
                pts, rsn = metric.score(info)
                total_score += pts
                reasons.extend(rsn)
                metric_breakdown.append({
                    "metric": metric.name,
                    "score": pts,
                    "max": metric.max_points,
                    "explanation": rsn,
                })
            result = {
                "symbol": ticker,
                "score": total_score,
                "max_score": self.max_score,
                "rating": self.get_rating(total_score),
                "reasons": reasons,
                "metrics": metric_breakdown,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "shortName": info.get("shortName"),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            logger.debug(f"Score result for {ticker}: {result}")
            return result
        except Exception as e:
            logger.error(f"Error scoring {ticker}: {e}")
            return {
                "symbol": ticker,
                "error": str(e),
                "score": 0,
                "max_score": self.max_score,
                "rating": "Error",
                "reasons": ["Failed to retrieve or process data"],
                "metrics": [],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def score_batch(self, tickers: List[str], parallel: bool = True, max_workers: int = 6) -> List[Dict[str, Any]]:
        if parallel and len(tickers) > 1:
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_ticker = {executor.submit(self.score_stock, t): t for t in tickers}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    res = future.result()
                    results.append(res)
            results.sort(key=lambda r: tickers.index(r["symbol"]) if r.get("symbol") in tickers else 9999)
            return results
        else:
            return [self.score_stock(ticker) for ticker in tickers]

    def explain_metrics(self) -> str:
        tbl = [["Metric", "Description", "Max Points"]]
        for m in self.metrics:
            tbl.append([m.name, m.description, m.max_points])
        return tbl

# =========================
# CLI and I/O Utilities
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Drovalix AI Stock Score Engine (Comprehensive Edition)"
    )
    parser.add_argument(
        "tickers", metavar="TICKER", type=str, nargs="*",
        help="Stock symbol(s) to score (comma or space separated, e.g. AAPL,MSFT)"
    )
    parser.add_argument(
        "-f", "--file", type=str, default=None,
        help="CSV/TXT file with tickers in first column"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output file for results (JSON, CSV, or Markdown)"
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Force CSV output (default: JSON)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Force JSON output (default: JSON)"
    )
    parser.add_argument(
        "--md", "--markdown", dest="markdown", action="store_true",
        help="Output in Markdown table format"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable detailed logging"
    )
    parser.add_argument(
        "--metrics", action="store_true",
        help="Show detailed metric breakdown for each stock"
    )
    parser.add_argument(
        "--explain-metrics", action="store_true",
        help="Show metric descriptions and exit"
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Batch score stocks in parallel (default: on for >2)"
    )
    parser.add_argument(
        "--max-workers", type=int, default=6,
        help="Max workers for parallel scoring (default: 6)"
    )
    return parser.parse_args()

def load_tickers_from_file(filepath: str) -> List[str]:
    tickers = []
    try:
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    tickers.append(row[0].strip())
        logger.info(f"Loaded {len(tickers)} tickers from {filepath}")
    except Exception as e:
        logger.error(f"Failed to load tickers from file {filepath}: {e}")
    return tickers

def save_results(results: List[Dict[str, Any]], output_file: str, as_csv: bool = False, as_md: bool = False, show_metrics: bool = False):
    try:
        if as_csv or output_file.endswith(".csv"):
            keys = ['symbol', 'score', 'max_score', 'rating', 'reasons', 'sector', 'industry', 'shortName']
            if show_metrics:
                keys.append("metrics")
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for res in results:
                    row = {k: res.get(k, "") for k in keys}
                    if isinstance(row['reasons'], list):
                        row['reasons'] = '; '.join(row['reasons'])
                    if show_metrics and isinstance(row.get('metrics'), list):
                        row['metrics'] = '; '.join(
                            [f"{m['metric']}={m['score']}/{m['max']}" for m in row['metrics']]
                        )
                    writer.writerow(row)
            logger.info(f"Results saved to {output_file} (CSV)")
        elif as_md or output_file.endswith(".md"):
            save_results_md(results, output_file, show_metrics=show_metrics)
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file} (JSON)")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def save_results_md(results: List[Dict[str, Any]], output_file: str, show_metrics: bool = False):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(results_md_table(results, show_metrics=show_metrics))
        logger.info(f"Results saved to {output_file} (Markdown)")
    except Exception as e:
        logger.error(f"Failed to save markdown: {e}")

def results_md_table(results: List[Dict[str, Any]], show_metrics: bool = False) -> str:
    if not results:
        return "No results."
    cols = ['Symbol', 'Score', 'Max', 'Rating', 'Sector', 'Industry', 'ShortName']
    if show_metrics:
        metric_names = [m['metric'] for m in results[0].get('metrics', [])]
        cols += metric_names
    header = '| ' + ' | '.join(cols) + ' |\n'
    header += '| ' + ' | '.join(['---'] * len(cols)) + ' |\n'
    rows = []
    for res in results:
        base = [
            res.get('symbol', ''),
            res.get('score', ''),
            res.get('max_score', ''),
            res.get('rating', ''),
            res.get('sector', ''),
            res.get('industry', ''),
            res.get('shortName', ''),
        ]
        if show_metrics and res.get('metrics'):
            base += [f"{m['score']}/{m['max']}" for m in res['metrics']]
        rows.append('| ' + ' | '.join(str(x) for x in base) + ' |')
    return header + '\n'.join(rows)

def print_results(results: List[Dict[str, Any]], as_csv: bool = False, as_md: bool = False, show_metrics: bool = False):
    if as_csv:
        keys = ['symbol', 'score', 'max_score', 'rating', 'sector', 'industry', 'shortName', 'reasons']
        if show_metrics:
            keys.append("metrics")
        writer = csv.DictWriter(sys.stdout, fieldnames=keys)
        writer.writeheader()
        for res in results:
            row = {k: res.get(k, "") for k in keys}
            if isinstance(row['reasons'], list):
                row['reasons'] = '; '.join(row['reasons'])
            if show_metrics and isinstance(row.get('metrics'), list):
                row['metrics'] = '; '.join(
                    [f"{m['metric']}={m['score']}/{m['max']}" for m in row['metrics']]
                )
            writer.writerow(row)
    elif as_md and PrettyTable:
        print(results_md_table(results, show_metrics=show_metrics))
    elif as_md:
        print(results_md_table(results, show_metrics=show_metrics))
    else:
        print(json.dumps(results, indent=2))

# =========================
# Main Entry Point
# =========================

def main():
    args = parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    scorer = DrovalixScorer()

    if args.explain_metrics:
        tbl = scorer.explain_metrics()
        if PrettyTable:
            pt = PrettyTable()
            pt.field_names = tbl[0]
            for row in tbl[1:]:
                pt.add_row(row)
            print(pt)
        else:
            print(results_md_table([{
                "metrics": [
                    {"metric": r[0], "score": r[2], "max": r[2], "explanation": r[1]}
                    for r in tbl[1:]
                ]
            }], show_metrics=True))
        sys.exit(0)

    tickers: List[str] = []
    if args.file:
        tickers = load_tickers_from_file(args.file)
    if args.tickers:
        for t in args.tickers:
            tickers.extend([tk.strip() for tk in t.split(',') if tk.strip()])
    tickers = list(dict.fromkeys([t.upper() for t in tickers if t]))
    if not tickers:
        logger.warning("No tickers provided. Defaulting to INFY.NS")
        tickers = ["INFY.NS"]

    use_parallel = args.parallel or (len(tickers) > 2)
    results = scorer.score_batch(tickers, parallel=use_parallel, max_workers=args.max_workers)

    if args.output:
        save_as_csv = args.csv or (args.output.endswith(".csv") and not args.json)
        save_as_md = args.markdown or args.output.endswith(".md")
        save_results(results, args.output, as_csv=save_as_csv, as_md=save_as_md, show_metrics=args.metrics)
    else:
        print_results(results, as_csv=args.csv, as_md=args.markdown, show_metrics=args.metrics)

if __name__ == "__main__":
    main()