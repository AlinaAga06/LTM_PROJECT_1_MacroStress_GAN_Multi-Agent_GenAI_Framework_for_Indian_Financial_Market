# LTM_PROJECT_1_MacroStress_GAN_Multi-Agent_GenAI_Framework_for_Indian_Financial_Market

A multi-agent Generative AI framework for financial stress testing, market scenario simulation, portfolio risk analysis, and portfolio optimization using Indian financial market data.

The system combines TimeGAN, LangGraph, Llama 3.1 8B, PyTorch, FastAPI, Streamlit, and PostgreSQL to simulate adverse market conditions and evaluate their impact on investment portfolios.

🎯 Project Objective

MacroStress-GAN is designed to:

Generate realistic financial time-series scenarios.

Simulate adverse market and macroeconomic conditions.

Estimate portfolio losses under stress scenarios.

Calculate VaR, Expected Shortfall, Expected Loss, and Maximum Drawdown.

Optimize portfolio allocation to reduce stress-induced losses.

Provide an interactive dashboard for stress-testing analysis.

Generate reports for financial risk analysis.

🏗️ System Workflow

Historical Market Data
        ↓
Data Processing & EDA
        ↓
Stress Scenario Generation
        ↓
TimeGAN Synthetic Time-Series
        ↓
Market Impact Analysis
        ↓
Portfolio Risk Analysis
        ↓
Portfolio Optimization
        ↓
Risk Report & Dashboard

🤖 Multi-Agent Architecture

The framework uses specialized agents for different stages of the workflow.

Agent

Responsibility

Orchestrator Agent

Coordinates the complete workflow

Scenario Agent

Creates market stress scenarios

Market Analysis Agent

Analyses market impact

Simulation Agent

Generates simulated market trajectories

Risk Agent

Calculates portfolio risk metrics

Critic / Validation Agent

Validates generated results

Report Agent

Produces final reports

📊 Market Variables

The current stock-market implementation uses five major financial variables:

NIFTY50

INDIA_VIX

CRUDE_OIL

USD_INR

INDIA_10Y_YIELD

Derived time-series features include:

NIFTY50_Return
CRUDE_OIL_Return
USD_INR_Return
INDIA_VIX_Change
INDIA_10Y_YIELD_Change

🧠 TimeGAN

The project uses TimeGAN to generate realistic synthetic financial time-series data while preserving important temporal and statistical relationships.

