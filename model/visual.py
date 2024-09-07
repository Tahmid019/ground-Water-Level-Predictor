import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

class DataVisualizer:
    def __init__(self, data):
        self.data = data
        self.processed_data = None
        
    def null_check(self):
        plt.figure(figsize=(10, 6))
        sb.heatmap(self.data.isnull(), cbar=False, cmap="viridis")
        plt.title("Heatmap of Null Values")
        plt.show()

    def count_plot(self, column_name):
        plt.figure(figsize=(8, 5))
        sb.countplot(x=column_name, data=self.data, palette='coolwarm')
        plt.title(f"Distribution of '{column_name}'")
        plt.show()

    def show_info(self):
        print("Dataset Info:")
        self.data.info()

    def correlation_heatmap(self):
        plt.figure(figsize=(10, 8))
        sb.heatmap(self.data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation Heatmap of Variables")
        plt.show()

    def plot_trends(self, columns, labels):
        plt.figure(figsize=(10, 6))
        for col, label in zip(columns, labels):
            self.data[col].plot(label=label)
        plt.legend()
        plt.title("Trends of Various Features")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.show()

    def plot_pie_chart(self, labels, sizes, colors):
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, startangle=90, shadow=True, 
                explode=(0, 0.01, 0.01, 0.01, 0.1, 0.2), autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
        plt.title("Groundwater Analysis Pie Chart")
        plt.show()

    def scatter_plot(self, x_col, y_col):
        sb.pairplot(self.data, x_vars=[x_col], y_vars=[y_col], kind='scatter', 
                    diag_kind='hist', height=5.0)
        plt.title(f"Scatter Plot between {x_col} and {y_col}")
        plt.show()


class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.processed_data = None

    def convert_categorical(self, column_name):
        availability = pd.get_dummies(self.data[column_name], drop_first=True)
        self.data.drop([column_name], axis=1, inplace=True)
        self.processed_data = pd.concat([self.data, availability], axis=1)
        return self.processed_data

    def get_mean_values(self):
        if self.processed_data is not None:
            return self.processed_data.mean()
        else:
            raise ValueError("Data has not been processed yet. Please convert categorical variables first.")


def main():
    data = pd.read_csv("data.csv")

    processor = DataProcessor(data)
    visualizer = DataVisualizer(data)

    visualizer.null_check()

    visualizer.count_plot('Situation')

    visualizer.show_info()

    processed_data = processor.convert_categorical('Situation')

    visualizer.correlation_heatmap()

    trend_columns = ["Total_Rainfall", "Net annual groundwater availability", "Total_Usage"]
    trend_labels = ["Total Rainfall", "Net Groundwater Availability", "Total Usage"]
    visualizer.plot_trends(trend_columns, trend_labels)

    pie_labels = ['Total Rainfall', 'Net Annual GroundWater', 'Total Use', 'Future Available',
                  'Projected demand (Domestic & Industrial, 2025)', 'Natural discharge (Non-monsoon)']
    pie_sizes = [14.84, 13.64, 8.39, 5.29, 1.063483, 1.210483]
    pie_colors = ['c', 'm', 'r', 'b', 'g', 'y']
    visualizer.plot_pie_chart(pie_labels, pie_sizes, pie_colors)

    visualizer.scatter_plot('Groundwater availability for future irrigation use', 'Net annual groundwater availability')

    mean_values = processor.get_mean_values()
    print("\nMean values of processed data:")
    print(mean_values)


if __name__ == "__main__":
    main()
