import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class TradingStrategy:
    """Backtest a trading strategy based on ML predictions"""
    
    def __init__(self, ticker, initial_capital=10000):
        self.ticker = ticker
        self.initial_capital = initial_capital
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        self.model = joblib.load(f'models/{self.ticker}_best_model.pkl')
        self.scaler = joblib.load(f'models/{self.ticker}_scaler.pkl')
        self.features = joblib.load(f'models/{self.ticker}_features.pkl')
    
    def backtest(self):
        """Run backtest on test data"""
        print(f"\n{'='*60}")
        print(f"BACKTESTING TRADING STRATEGY - {self.ticker}")
        print(f"{'='*60}")
        
        # Load test data
        test_df = pd.read_csv(f'data/{self.ticker}_test.csv', index_col='Date', parse_dates=True)
        
        X_test = test_df[self.features]
        X_scaled = self.scaler.transform(X_test)
        
        # Get predictions
        predictions = self.model.predict(X_scaled)
        actual_prices = test_df['Close'].values
        
        # Trading logic
        capital = self.initial_capital
        shares = 0
        portfolio_values = []
        buy_hold_values = []
        trades = []
        
        # Buy and hold baseline
        buy_hold_shares = self.initial_capital / actual_prices[0]
        
        for i in range(len(actual_prices) - 1):
            current_price = actual_prices[i]
            next_pred = predictions[i]
            next_actual = actual_prices[i + 1]
            
            # Buy and Hold value
            buy_hold_value = buy_hold_shares * current_price
            buy_hold_values.append(buy_hold_value)
            
            # ML Strategy
            # If we predict price will go up and we have no shares, BUY
            if next_pred > current_price and shares == 0 and capital > 0:
                shares = capital / current_price
                capital = 0
                trades.append(('BUY', test_df.index[i], current_price, shares))
            
            # If we predict price will go down and we have shares, SELL
            elif next_pred < current_price and shares > 0:
                capital = shares * current_price
                shares = 0
                trades.append(('SELL', test_df.index[i], current_price, capital))
            
            # Calculate current portfolio value
            portfolio_value = capital + (shares * current_price)
            portfolio_values.append(portfolio_value)
        
        # Final settlement
        if shares > 0:
            final_capital = shares * actual_prices[-1]
        else:
            final_capital = capital
        
        portfolio_values.append(final_capital)
        buy_hold_final = buy_hold_shares * actual_prices[-1]
        buy_hold_values.append(buy_hold_final)
        
        # Calculate returns
        strategy_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        buy_hold_return = ((buy_hold_final - self.initial_capital) / self.initial_capital) * 100
        outperformance = strategy_return - buy_hold_return
        
        # Calculate max drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdown = (running_max - portfolio_values) / running_max * 100
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe Ratio (assuming 0% risk-free rate)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Print results
        print(f"\n💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📅 Test Period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
        print(f"📊 Number of Trading Days: {len(actual_prices)}")
        print(f"🔄 Number of Trades: {len(trades)}")
        
        print(f"\n{'='*60}")
        print("STRATEGY PERFORMANCE")
        print(f"{'='*60}")
        print(f"  Final Portfolio Value: ${final_capital:,.2f}")
        print(f"  Total Return: {strategy_return:+.2f}%")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print(f"\n{'='*60}")
        print("BUY & HOLD BASELINE")
        print(f"{'='*60}")
        print(f"  Final Portfolio Value: ${buy_hold_final:,.2f}")
        print(f"  Total Return: {buy_hold_return:+.2f}%")
        
        print(f"\n{'='*60}")
        if outperformance > 0:
            print(f"🎉 OUTPERFORMANCE: +{outperformance:.2f}%")
            print("✅ ML Strategy BEAT Buy & Hold!")
        else:
            print(f"📉 UNDERPERFORMANCE: {outperformance:.2f}%")
            print("❌ ML Strategy did not beat Buy & Hold")
        print(f"{'='*60}")
        
        # Show recent trades
        if trades:
            print(f"\n📋 Last 5 Trades:")
            for trade in trades[-5:]:
                action, date, price, amount = trade
                if action == 'BUY':
                    print(f"  🟢 BUY  - {date.date()} @ ${price:.2f} ({amount:.2f} shares)")
                else:
                    print(f"  🔴 SELL - {date.date()} @ ${price:.2f} (${amount:.2f} capital)")
        
        # Plot results
        self.plot_backtest(test_df.index[:len(portfolio_values)], 
                          portfolio_values, buy_hold_values, trades, test_df)
        
        return {
            'ticker': self.ticker,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'outperformance': outperformance,
            'num_trades': len(trades),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_value': final_capital
        }
    
    def plot_backtest(self, dates, strategy_values, buy_hold_values, trades, test_df):
        """Create backtest visualization"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # Portfolio value over time
        ax1.plot(dates, strategy_values, label='ML Strategy', linewidth=2.5, color='#2E86AB')
        ax1.plot(dates, buy_hold_values, label='Buy & Hold', linewidth=2.5, color='#F18F01', linestyle='--', alpha=0.8)
        ax1.axhline(y=self.initial_capital, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='Initial Capital')
        
        # Mark buy/sell points
        for trade in trades:
            action, date, price, _ = trade
            if date in dates:
                idx = dates.get_loc(date)
                if action == 'BUY':
                    ax1.scatter(date, strategy_values[idx], color='green', s=100, marker='^', zorder=5, label='Buy' if trade == trades[0] else '')
                else:
                    ax1.scatter(date, strategy_values[idx], color='red', s=100, marker='v', zorder=5, label='Sell' if trade == trades[0] else '')
        
        ax1.set_title(f'{self.ticker} - Trading Strategy Backtest Results', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Stock price
        ax2.plot(test_df.index, test_df['Close'], linewidth=2, color='#A23B72', label='Stock Price')
        ax2.set_title(f'{self.ticker} Stock Price During Backtest Period', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'plots/{self.ticker}_backtest.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ Backtest chart saved: plots/{self.ticker}_backtest.png")
        plt.close()

if __name__ == "__main__":
    print("="*60)
    print("STOCK TRADING STRATEGY BACKTEST")
    print("="*60)
    
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    results = []
    
    for ticker in tickers:
        strategy = TradingStrategy(ticker, initial_capital=10000)
        result = strategy.backtest()
        results.append(result)
    
    # Summary comparison
    print("\n\n" + "="*60)
    print("BACKTEST SUMMARY - ALL STOCKS")
    print("="*60)
    
    summary_df = pd.DataFrame(results)
    summary_df = summary_df[['ticker', 'strategy_return', 'buy_hold_return', 'outperformance', 
                             'num_trades', 'max_drawdown', 'sharpe_ratio', 'final_value']]
    
    print("\n" + summary_df.to_string(index=False))
    
    # Best performer
    best = summary_df.loc[summary_df['outperformance'].idxmax()]
    print(f"\n🏆 Best Performer: {best['ticker']} with {best['outperformance']:+.2f}% outperformance")
    
    print("\n" + "="*60)
    print("🎉 BACKTEST COMPLETE!")
    print("="*60)