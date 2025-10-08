'''
Midterm Project!
DS 2023
Design 1: Communicating with Data

Research Questions:
    1. How do ESG Scores Range across different industries?
    2. How do ESG Scores of Mag Seven Companies compare to aggregate ESG score of other S&P Companies?
'''


# 3. Make Data Visualizations

# Density Plot of ESG Scores across different Industries
plt.figure(figsize=(8,5))
sns.kdeplot(esg, x='totalEsg', hue='GICS Sector', fill=True, alpha=0.03)
plt.title("ESG Scores Across Industries")
plt.xlabel("ESG Score")
plt.savefig("./plots/esg_density_plot.png")
#plt.show()
plt.close()


# 3D Scatterplot of 3 ESG Factors
fig_3D = px.scatter_3d(esg, x="environmentScore", y='socialScore', z='governanceScore', 
                        color='GICS Sector', hover_data=['Symbol'], title="The Three Components of ESG across the S&P 500", 
                        labels={"environmentScore": "Environment", 'socialScore': "Social", 'governanceScore': "Governance"})
fig_3D.update_traces(marker=dict(line=dict(width=2, color='White')))
fig_3D.write_html("./html/interactive_3d_esg.html")
fig_3D.show()


# Mag Seven ESG Scores Compared to Others
esg_columns = ['environmentScore', 'socialScore', 'governanceScore', 'totalEsg']
esg_table = esg_filtered.groupby("Symbol")[esg_columns].mean().sort_values(by='totalEsg', ascending=False)
print(esg_table)


# Mag Seven Stacked Bar Chart
esg_table = esg_table[['environmentScore', 'socialScore', 'governanceScore']]
esg_table.plot(kind='bar', stacked=True, figsize=(10,6))
plt.title("ESG of the Magnificent Seven")
plt.ylabel("ESG Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(labels=["Environmental", "Social", "Governance"])
plt.savefig("./plots/mag7_esg_bar.png")
#plt.show()
plt.close()


# Time Series of S&P Data
fig = px.line(sp500, x = "Date", y='Value',
              title = 'S&P 500 Index')
fig.update_traces(line=dict(width=2), line_color = 'green')
fig.update_layout(xaxis_showgrid=True, yaxis_showgrid=False, plot_bgcolor='white', 
                  xaxis=dict(gridcolor='lightgrey'), yaxis=dict(gridcolor='lightgrey'),
                  title_font=dict(color='black', size=30, family="Retina"))
fig.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1))
fig.write_html("./html/s&p_timeseries.html")
fig.show()


# Mag7 and GDP DataFrame and Barplot
value_dict = {"Mag 7": 17.3,
              "S&P 500" : 46,
              "USA": 30.34,
              "China": 19.53,
              "Canada": 2.33,
              "Germany": 4.53}

value_df = pd.DataFrame(list(value_dict.items()), columns=['Country', 'Value']).sort_values(by='Value', ascending=False)

sns.barplot(value_df, x='Country', y='Value', palette='viridis')
plt.title("GDP Compared to Market Value")
plt.xlabel("Country (GDP)")
plt.ylabel("Value (Trillions)")
plt.legend(title="Value in Trillions of Dollars")
plt.xticks(rotation=30)
plt.savefig("./plots/economy_bar.png")
#plt.show()
plt.close()


# Time Series of Mag 7 Performance
mag7_2023 = mag7[mag7['Date'] > '2022-12-31']

fig2 = px.line(mag7_2023, x = "Date", y = "Close", color = 'Symbol', 
               title='Magnificent Seven Stock Performance')
fig2.update_layout(xaxis_showgrid=True, yaxis_showgrid=False, plot_bgcolor='white', 
                  xaxis=dict(gridcolor='lightgrey'), yaxis=dict(gridcolor='lightgrey'),
                  title_font=dict(color='black', size=24))
fig2.add_shape(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1, line=dict(color='black', width=1))
fig2.write_html("./html/mag7_timeseries.html")
fig2.show()





## Extra Plots

# Swarm Plot of Mag Seven compared to other Stocks
sns.swarmplot(data=esg_filtered[esg_filtered['Symbol'] == 'Other'], x='totalEsg', hue='Symbol', dodge=True, color='0.5', zorder=1)
for stock in mag7_list:
    stock_data = esg_filtered[esg_filtered['Symbol'] == stock]
    plt.scatter(stock_data['totalEsg'], [0] * len(stock_data), label=stock, s=300, zorder=2)
plt.legend(bbox_to_anchor = (1, 1))
#plt.show()
#plt.savefig("./plots/esg_comparison_swarm.png")
plt.close()

# KDE Plots of Mag Seven and Volume
sns.kdeplot(mag7, x="Volume", hue="Symbol", fill=True, alpha=0.3)
plt.xlim(0, 0.5E9)
#plt.show()
plt.close()

# KDE Plot of Mag Seven and Volume (log)
sns.kdeplot(mag7, x="Volume", hue="Symbol", fill=True, alpha=0.3)
plt.xscale("log")
plt.xlabel("Volume (log)")
#plt.show()
#plt.savefig("./plots/mag7_density_log.png")
plt.close()


sns.boxplot(esg_filtered[esg_filtered['Symbol'] == 'Other'], x='totalEsg', hue='Symbol', color='green')
plt.title("ESG Scores Across S&P 500")
plt.xlabel("ESG Score")
#plt.show()
#plt.savefig("./plots/esg_box.png")
plt.close()

