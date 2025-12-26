#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import google.generativeai as genai
from datetime import datetime
import time
import os
import json
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ────────────────────────────────
# 🔑 GEMINI API CONFIGURATION
# ────────────────────────────────
GEMINI_API_KEY = "enteer your key"
genai.configure(api_key=GEMINI_API_KEY)

# ────────────────────────────────
# 🎨 PAGE CONFIGURATION
# ────────────────────────────────
st.set_page_config(
    page_title="FinBot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────
# 💾 SESSION STATE INITIALIZATION
# ────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model" not in st.session_state:
    st.session_state.model = None
if "ai_enabled" not in st.session_state:
    st.session_state.ai_enabled = False
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ────────────────────────────────
# 🛠️ TOOL FUNCTIONS FOR AI AGENT
# ────────────────────────────────

def get_stock_price(ticker: str) -> dict:
    """Fetch current stock price and basic info"""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        hist = stock.history(period="1d")
        
        return {
            "success": True,
            "ticker": ticker.upper(),
            "price": round(hist['Close'].iloc[-1], 2) if not hist.empty else info.get('currentPrice', 'N/A'),
            "change": round(hist['Close'].iloc[-1] - hist['Open'].iloc[-1], 2) if not hist.empty else 'N/A',
            "company_name": info.get('longName', ticker.upper()),
            "market_cap": info.get('marketCap', 'N/A'),
            "pe_ratio": info.get('trailingPE', 'N/A'),
            "52w_high": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52w_low": info.get('fiftyTwoWeekLow', 'N/A')
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_stock_history(ticker: str, period: str = "1mo") -> dict:
    """Fetch historical stock data"""
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)
        
        return {
            "success": True,
            "ticker": ticker.upper(),
            "data": hist,
            "period": period
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def plot_stock_chart(ticker: str, period: str = "1mo"):
    """Generate interactive stock price chart"""
    try:
        result = get_stock_history(ticker, period)
        if not result["success"]:
            return None
        
        df = result["data"]
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker.upper()
        ))
        
        # Add volume bar chart
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color='rgba(100, 149, 237, 0.3)'
        ))
        
        fig.update_layout(
            title=f'{ticker.upper()} Stock Price - {period}',
            yaxis_title='Price ($)',
            yaxis2=dict(
                title='Volume',
                overlaying='y',
                side='right'
            ),
            xaxis_title='Date',
            template='plotly_dark',
            hovermode='x unified',
            height=500
        )
        
        return fig
    except Exception as e:
        return None

def add_to_portfolio(ticker: str, shares: float, buy_price: float) -> dict:
    """Add stock to portfolio"""
    try:
        ticker = ticker.upper()
        if ticker in st.session_state.portfolio:
            # Update existing position
            old_shares = st.session_state.portfolio[ticker]['shares']
            old_avg = st.session_state.portfolio[ticker]['avg_price']
            new_shares = old_shares + shares
            new_avg = ((old_shares * old_avg) + (shares * buy_price)) / new_shares
            
            st.session_state.portfolio[ticker] = {
                'shares': new_shares,
                'avg_price': round(new_avg, 2),
                'added_date': st.session_state.portfolio[ticker]['added_date']
            }
        else:
            st.session_state.portfolio[ticker] = {
                'shares': shares,
                'avg_price': buy_price,
                'added_date': datetime.now().strftime("%Y-%m-%d")
            }
        
        return {
            "success": True,
            "message": f"Added {shares} shares of {ticker} at ${buy_price}",
            "portfolio": st.session_state.portfolio
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_portfolio() -> dict:
    """Analyze current portfolio performance"""
    try:
        if not st.session_state.portfolio:
            return {"success": False, "error": "Portfolio is empty"}
        
        portfolio_data = []
        total_value = 0
        total_cost = 0
        
        for ticker, position in st.session_state.portfolio.items():
            current_price_data = get_stock_price(ticker)
            if current_price_data["success"]:
                current_price = current_price_data["price"]
                shares = position['shares']
                avg_price = position['avg_price']
                
                current_value = current_price * shares
                cost_basis = avg_price * shares
                gain_loss = current_value - cost_basis
                gain_loss_pct = (gain_loss / cost_basis) * 100
                
                portfolio_data.append({
                    'ticker': ticker,
                    'shares': shares,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'current_value': round(current_value, 2),
                    'cost_basis': round(cost_basis, 2),
                    'gain_loss': round(gain_loss, 2),
                    'gain_loss_pct': round(gain_loss_pct, 2)
                })
                
                total_value += current_value
                total_cost += cost_basis
        
        total_gain_loss = total_value - total_cost
        total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            "success": True,
            "holdings": portfolio_data,
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_gain_loss": round(total_gain_loss, 2),
            "total_gain_loss_pct": round(total_gain_loss_pct, 2)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def compare_stocks(tickers: list) -> dict:
    """Compare multiple stocks"""
    try:
        comparison_data = []
        for ticker in tickers:
            data = get_stock_price(ticker)
            if data["success"]:
                comparison_data.append(data)
        
        return {
            "success": True,
            "comparison": comparison_data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ────────────────────────────────
# 🤖 ENHANCED AI AGENT WITH TOOLS
# ────────────────────────────────

SYSTEM_PROMPT = """You are FinanceGPT, an advanced AI financial advisor and agent. You have access to real-time stock market data and portfolio management tools.

Your capabilities include:
1. Fetching real-time stock prices and company information
2. Analyzing historical stock performance
3. Managing user portfolios (adding positions, tracking performance)
4. Comparing multiple stocks
5. Providing financial advice and explanations

When users ask about specific stocks, automatically use your tools to fetch real data. When they want to track stocks, add them to their portfolio. Always provide data-driven insights.

Available tools:
- get_stock_price(ticker): Get current price and info
- get_stock_history(ticker, period): Get historical data
- add_to_portfolio(ticker, shares, buy_price): Add to portfolio
- analyze_portfolio(): Analyze portfolio performance
- compare_stocks(tickers): Compare multiple stocks

Be proactive in using these tools to provide accurate, real-time information."""

def initialize_ai_agent():
    """Initialize the enhanced Gemini AI agent"""
    try:
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 4096,
            },
            system_instruction=SYSTEM_PROMPT
        )
        st.session_state.model = model
        st.session_state.ai_enabled = True
        return True
    except Exception as e:
        st.error(f"Failed to initialize AI: {str(e)}")
        return False

def process_user_query(query: str) -> str:
    """Process user query with tool calling capability"""
    
    # Detect intent and call appropriate tools
    query_lower = query.lower()
    context_data = ""
    
    # Stock price queries
    if any(word in query_lower for word in ['price', 'stock', 'quote', 'trading at']):
        words = query.split()
        for word in words:
            if word.isupper() and len(word) <= 5:  # Likely a ticker
                stock_data = get_stock_price(word)
                if stock_data["success"]:
                    context_data += f"\n\nReal-time data for {stock_data['ticker']}:\n"
                    context_data += f"Current Price: ${stock_data['price']}\n"
                    context_data += f"Company: {stock_data['company_name']}\n"
                    context_data += f"Change: ${stock_data['change']}\n"
                    context_data += f"Market Cap: {stock_data['market_cap']}\n"
                    context_data += f"P/E Ratio: {stock_data['pe_ratio']}\n"
    
    # Portfolio analysis
    if any(word in query_lower for word in ['portfolio', 'holdings', 'my stocks', 'my investments']):
        portfolio_analysis = analyze_portfolio()
        if portfolio_analysis["success"]:
            context_data += f"\n\nYour Portfolio Analysis:\n"
            context_data += f"Total Value: ${portfolio_analysis['total_value']}\n"
            context_data += f"Total Cost: ${portfolio_analysis['total_cost']}\n"
            context_data += f"Total Gain/Loss: ${portfolio_analysis['total_gain_loss']} ({portfolio_analysis['total_gain_loss_pct']}%)\n"
            context_data += f"\nHoldings:\n"
            for holding in portfolio_analysis['holdings']:
                context_data += f"- {holding['ticker']}: {holding['shares']} shares @ ${holding['avg_price']} (Current: ${holding['current_price']}, P/L: ${holding['gain_loss']})\n"
    
    # Create enhanced prompt
    enhanced_query = query
    if context_data:
        enhanced_query = f"{query}\n\n[Real-time Market Data]:{context_data}\n\nProvide analysis based on this current data."
    
    return enhanced_query

# Initialize the model
if not st.session_state.ai_enabled:
    initialize_ai_agent()

# ────────────────────────────────
# 🎨 SIDEBAR
# ────────────────────────────────
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("---")

    if st.session_state.ai_enabled:
        st.markdown('<div style="color: #00FF00; font-weight: bold;">🟢 AI Agent Active</div>', unsafe_allow_html=True)
        st.caption("Real-time market data enabled")
    else:
        st.markdown('<div style="color: red; font-weight: bold;">🔴 AI Disconnected</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Quick Actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Analyze Portfolio"):
            analysis = analyze_portfolio()
            if analysis["success"]:
                st.success(f"Total P/L: ${analysis['total_gain_loss']}")
            else:
                st.warning("Portfolio is empty")
    
    with col2:
        if st.button("🔄 Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Stock Chart Viewer
    with st.expander("📈 View Stock Chart"):
        chart_ticker = st.text_input("Enter Ticker", key="chart_ticker", placeholder="AAPL")
        chart_period = st.selectbox("Time Period", 
                                     ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y"],
                                     index=2)
        
        if st.button("Show Chart") and chart_ticker:
            fig = plot_stock_chart(chart_ticker, chart_period)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not fetch chart data")
    
    st.markdown("---")
    
    # Add to Portfolio
    with st.expander("➕ Add to Portfolio"):
        ticker = st.text_input("Ticker Symbol", key="add_ticker")
        shares = st.number_input("Shares", min_value=0.01, value=1.0, step=0.01)
        buy_price = st.number_input("Buy Price ($)", min_value=0.01, value=100.0, step=0.01)
        
        if st.button("Add Position"):
            if ticker:
                result = add_to_portfolio(ticker, shares, buy_price)
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["error"])
    
    st.markdown("---")
    
    # Portfolio Summary
    st.subheader("💼 Portfolio Summary")
    if st.session_state.portfolio:
        for ticker, position in st.session_state.portfolio.items():
            st.markdown(f"**{ticker}**: {position['shares']} shares @ ${position['avg_price']}")
    else:
        st.info("No positions yet")
    
    st.markdown("---")
    st.subheader("📖 About")
    st.info("FinanceGPT is an AI agent with real-time market data, portfolio tracking, and intelligent financial analysis.")

# ────────────────────────────────
# 💬 MAIN CHAT AREA
# ────────────────────────────────
st.title("💰 FinanceGPT — Your AI Financial Agent")
st.caption("🤖 Powered by real-time market data | Ask about stocks, analyze portfolios, or get financial advice!")

# Display chat history
for chat in st.session_state.chat_history:
    st.chat_message("user").markdown(chat['user'])
    st.chat_message("assistant").markdown(chat['bot'])

# Chat input
user_input = st.chat_input("Ask me anything about finance, stocks, or your portfolio...")

if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)

    if st.session_state.ai_enabled and st.session_state.model:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing with real-time data..."):
                try:
                    # Process query with tools
                    enhanced_query = process_user_query(user_input)
                    
                    # Generate response
                    response = st.session_state.model.generate_content(enhanced_query)
                    bot_reply = response.text
                    
                except Exception as e:
                    bot_reply = f"⚠️ Error: {str(e)}"
            
            st.markdown(bot_reply)

        # Save to history
        st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
    else:
        st.warning("AI model not initialized. Please refresh the page.")

# ────────────────────────────────
# 📊 DASHBOARD (OPTIONAL)
# ────────────────────────────────
if st.session_state.portfolio:
    st.markdown("---")
    st.subheader("📊 Portfolio Dashboard")
    
    analysis = analyze_portfolio()
    if analysis["success"]:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${analysis['total_value']:,.2f}")
        with col2:
            st.metric("Total Cost", f"${analysis['total_cost']:,.2f}")
        with col3:
            st.metric("Total P/L", f"${analysis['total_gain_loss']:,.2f}", 
                     delta=f"{analysis['total_gain_loss_pct']:.2f}%")
        with col4:
            st.metric("Positions", len(analysis['holdings']))
        
        # Holdings table
        df = pd.DataFrame(analysis['holdings'])
        st.dataframe(df, use_container_width=True)
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Portfolio Allocation Pie Chart
            st.markdown("#### 📊 Portfolio Allocation")
            fig_pie = px.pie(
                df, 
                values='current_value', 
                names='ticker',
                title='Distribution by Value',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with chart_col2:
            # Gain/Loss Bar Chart
            st.markdown("#### 📈 Gain/Loss by Position")
            colors = ['#00ff00' if x > 0 else '#ff4444' for x in df['gain_loss']]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=df['ticker'],
                y=df['gain_loss'],
                marker_color=colors,
                text=df['gain_loss'].apply(lambda x: f"${x:.2f}"),
                textposition='outside'
            ))
            fig_bar.update_layout(
                title='Profit/Loss by Stock',
                xaxis_title='Stock Ticker',
                yaxis_title='Gain/Loss ($)',
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Performance comparison chart
        st.markdown("#### 📉 Return on Investment Comparison")
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            x=df['ticker'],
            y=df['gain_loss_pct'],
            marker_color=df['gain_loss_pct'].apply(
                lambda x: '#00ff00' if x > 0 else '#ff4444'
            ),
            text=df['gain_loss_pct'].apply(lambda x: f"{x:.2f}%"),
            textposition='outside'
        ))
        fig_roi.update_layout(
            title='ROI % by Position',
            xaxis_title='Stock Ticker',
            yaxis_title='Return on Investment (%)',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_roi, use_container_width=True)

# ────────────────────────────────
# 📈 STOCK COMPARISON TOOL
# ────────────────────────────────
st.markdown("---")
st.subheader("🔍 Compare Stocks")

compare_col1, compare_col2 = st.columns([3, 1])

with compare_col1:
    compare_tickers = st.text_input(
        "Enter tickers separated by commas (e.g., AAPL, GOOGL, MSFT)",
        placeholder="AAPL, GOOGL, MSFT"
    )

with compare_col2:
    compare_period = st.selectbox(
        "Period",
        ["1mo", "3mo", "6mo", "1y", "5y"],
        key="compare_period"
    )

if st.button("📊 Compare Stocks") and compare_tickers:
    tickers_list = [t.strip().upper() for t in compare_tickers.split(",")]
    
    with st.spinner("Fetching data..."):
        # Create comparison chart
        fig_compare = go.Figure()
        
        for ticker in tickers_list:
            result = get_stock_history(ticker, compare_period)
            if result["success"]:
                df = result["data"]
                # Normalize to percentage change from start
                normalized = ((df['Close'] / df['Close'].iloc[0]) - 1) * 100
                fig_compare.add_trace(go.Scatter(
                    x=df.index,
                    y=normalized,
                    mode='lines',
                    name=ticker,
                    line=dict(width=2)
                ))
        
        fig_compare.update_layout(
            title=f'Stock Performance Comparison - {compare_period}',
            xaxis_title='Date',
            yaxis_title='Return (%)',
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
        
        # Show comparison table
        comparison_data = []
        for ticker in tickers_list:
            stock_info = get_stock_price(ticker)
            if stock_info["success"]:
                comparison_data.append({
                    'Ticker': ticker,
                    'Price': f"${stock_info['price']}",
                    'Change': f"${stock_info['change']}",
                    'P/E Ratio': stock_info['pe_ratio'],
                    '52W High': f"${stock_info['52w_high']}" if stock_info['52w_high'] != 'N/A' else 'N/A',
                    '52W Low': f"${stock_info['52w_low']}" if stock_info['52w_low'] != 'N/A' else 'N/A'
                })
        
        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)


# In[ ]:




