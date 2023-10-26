import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import warnings

class M6Project:
    def __init__(self, data_path):
        """
        Initialize the M6Project instance.

        Read the data file and convert it to a DataFrame.

        Parameters
        ----------
        data_path (str): Path to the data file.
        """
        self.df = pd.read_csv(data_path, delimiter='\t', names=["Origin", "Destination", "Origin City","Destination City","Passengers","Seats","Flights","Distance","Fly Date","Origin Population","Destination Population"])
        print(self.df)

    def data_wrangling(self):
        """
        Perform data wrangling operations on the DataFrame.

        This method modifies the DataFrame in place.

        Returns
        -------
        Top 5 rows data
        """
        # Split the 'Origin' column into 'Origin City' and 'Origin State' columns
        self.df[['Origin_City', 'Origin_State']] = self.df['Origin City'].str.split(', ', expand=True)

        # Drop the original "Origin City" column if you no longer need it
        self.df = self.df.drop(columns=['Origin City'])
        
        # Split the 'Destination' column into 'Destination City' and 'Destination State' columns
        self.df[['Destination_City', 'Destination_State']] = self.df['Destination City'].str.split(', ', expand=True)

        # Drop the original 'Destination City' column if needed
        self.df = self.df.drop(columns=['Destination City'])
        
        # Convert the "Fly Date" column to a string
        self.df['Fly Date'] = self.df['Fly Date'].astype(str)

        # Extract year and month using string slicing
        self.df['Year'] = self.df['Fly Date'].str[:4]
        self.df['Month'] = self.df['Fly Date'].str[4:]

        # Convert the new columns to integers if needed
        self.df['Year'] = self.df['Year'].astype(int)
        self.df['Month'] = self.df['Month'].astype(int)

        # Drop the 'Fly Date' column
        self.df = self.df.drop(columns=['Fly Date'])
        print(self.df.head())

    def desired_order(self):
        """
        Rearrange DataFrame columns based on a desired order.

        Returns
        -------
        None
        """
        # Define the desired column order
        desired_order = [
            'Origin', 'Origin_City', 'Origin_State',
            'Destination', 'Destination_City', 'Destination_State',
            'Year', 'Month',
            'Passengers', 'Seats', 'Flights', 'Distance',
            'Origin Population', 'Destination Population'
        ]

        # Reorder the DataFrame columns
        self.df = self.df[desired_order]
        print(self.df.head())

    def shape(self):
        """
        Get the shape and column data types of the DataFrame.

        Returns
        -------
        None
        """
        # Get the number of rows and columns in the dataset
        num_rows, num_columns = self.df.shape
        print(f"Number of Rows: {num_rows}")
        print(f"Number of Columns: {num_columns}")

        # List the column names and their data types
        column_data_types = self.df.dtypes
        print("Column Data Types:")
        print(column_data_types)

    def summary(self):
        """
        Display summary statistics for the DataFrame.

        Returns
        -------
        None
        """
        # Set the float format for display
        pd.set_option('display.float_format', '{:.1f}'.format)

        # Get summary statistics
        summary_stats = self.df.describe()
        print(summary_stats)

    def unique_values(self):
        """
        Find unique values in categorical columns.

        This method identifies and prints unique values in non-numeric (categorical) columns of the DataFrame.

        Returns
        -------
        None
        """
        categorical_columns = self.df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            unique_values = self.df[column].unique()
            print(f"Unique values in '{column}': {unique_values}")

    def passenger_trends(self):
        """
        Plot trends in passenger numbers over the years.

        This method visualizes the trends in passenger numbers over the years using a line plot.

        Returns
        -------
        None
        """
        # Assuming 'Year' and 'Passengers' are columns in your DataFrame
        passenger_trends = self.df.groupby('Year')['Passengers'].sum()
        passenger_trends.plot(kind='line')
        plt.title('Trends in Passenger Numbers Over the Years')
        plt.xlabel('Year')
        plt.ylabel('Total Passengers')
        plt.xticks(range(1990, 2010, 2))  # Set the x-axis ticks to show integers from 1990 to 2009 with a step of 2
        formatter = mtick.FuncFormatter(lambda x, _: f'{int(x / 1e6):,}M')
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.show()

    def passenger_trends_seaborn(self):
        """
        Plot trends in passenger numbers over the years using Seaborn.

        This method visualizes the trends in passenger numbers over the years using a Seaborn line plot.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        sns.set(style="whitegrid", font_scale=1.2, palette="Set1")
        passenger_trends = self.df.groupby('Year')['Passengers'].sum().reset_index()
        plot = sns.lineplot(data=passenger_trends, x='Year', y='Passengers', marker='o')
        plt.title('Trends in Passenger Numbers Over the Years', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Total Passengers (in millions)', fontsize=12)
        plt.xticks(range(1990, 2010, 2))
        formatter = mtick.FuncFormatter(lambda x, _: f'{int(x / 1e6):,}M')
        plot.yaxis.set_major_formatter(formatter)
        plt.show()

    def top_routes(self):
        """
        Find and print the top routes by total passengers.

        This method identifies and prints the top routes based on the total number of passengers.

        Returns
        -------
        None
        """
        # Combining Origin and Destination columns to create a 'Route' column
        self.df['Route'] = self.df['Origin'] + ' to ' + self.df['Destination']
        top_routes = self.df.groupby('Route')['Passengers'].sum().sort_values(ascending=False)
        print(top_routes)

    def top_city_routes(self):
        """
        Find and print the top city-to-city routes by total passengers.

        This method identifies and prints the top city-to-city routes based on the total number of passengers.

        Returns
        -------
        None
        """
        # Combining Origin City and Destination City columns to create a 'City_Route' column
        self.df['City_Route'] = self.df['Origin_City'] + ' to ' + self.df['Destination_City']
        top_city_routes = self.df.groupby('City_Route')['Passengers'].sum().sort_values(ascending=False)
        print(top_city_routes)

    def top_state_routes(self):
        """
        Find and print the top state-to-state routes by total passengers.

        This method identifies and prints the top state-to-state routes based on the total number of passengers.

        Returns
        -------
        None
        """
        # Combining Origin State and Destination State columns to create a 'State_Route' column
        self.df['State_Route'] = self.df['Origin_State'] + ' to ' + self.df['Destination_State']
        top_state_routes = self.df.groupby('State_Route')['Passengers'].sum().sort_values(ascending=False)
        print(top_state_routes)

    def monthly_passengers(self):
        """
        Calculate and return the monthly passenger trends.

        Returns
        -------
        pandas.Series
            Monthly passenger trends.
        """
        monthly_passengers = self.df.groupby('Month')['Passengers'].mean().sort_values(ascending=False)
        return monthly_passengers

    def monthly_passengers_plot(self):
        """
        Create a line plot of monthly passenger trends.

        Returns
        -------
        None
        """
        monthly_passengers = self.df.groupby(['Year', 'Month'])['Passengers'].mean().unstack().T
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the data for all years
        for year in range(1990, 2010):
            plt.plot(monthly_passengers.index, monthly_passengers[year], label=str(year))

        # Set plot labels and legend
        plt.title('Monthly Passenger Trends (1990-2009)')
        plt.xlabel('Month')
        plt.ylabel('Average Passengers')
        plt.xticks(monthly_passengers.index)
        plt.legend(title='Year', loc='upper right')

        # Show the plot
        plt.show()

    def monthly_passengers_seaborn(self):
        """
        Create a Seaborn line plot of monthly passenger trends.

        Returns
        -------
        None
        """
        # Calculate and compare monthly passenger trends for all years
        monthly_passengers = self.df.groupby(['Year', 'Month'])['Passengers'].mean().unstack().T

        # Set the Seaborn style
        sns.set(style='whitegrid')

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the data for all years using Seaborn
        for year in range(1990, 2010):
            sns.lineplot(data=monthly_passengers, x=monthly_passengers.index, y=year, label=str(year))

        # Set plot labels and legend
        plt.title('Monthly Passenger Trends (1990-2009)')
        plt.xlabel('Month')
        plt.ylabel('Average Passengers')

        # Create a legend outside of the plot
        ax.legend(title='Year', loc='center left', bbox_to_anchor=(1, 0.5))

        # Show the plot
        plt.show()

    def flights_per_route(self):
        """
        Calculate and return the number of flights per route.

        Returns
        -------
        pandas.Series
            Number of flights per route.
        """
        flights_per_route = self.df['Route'].value_counts()
        return flights_per_route

    def distance_vs_passengers(self):
        """
        Create a scatter plot of distance vs. passengers.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Distance'], self.df['Passengers'], alpha=0.5)
        plt.title('Distance vs. Passengers Scatter Plot')
        plt.xlabel('Distance')
        plt.ylabel('Passengers')
        plt.show()

    def distance_vs_passengers_seaborn(self):
        """
        Create a Seaborn scatter plot of distance vs. passengers.

        Returns
        -------
        None
        """
        sns.set(style='whitegrid', context='notebook', palette='dark')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='Distance', y='Passengers', alpha=0.5)
        plt.xlabel('Distance')
        plt.ylabel('Passengers')
        plt.title('Distance vs. Passengers Scatter Plot')
        sns.regplot(data=self.df, x='Distance', y='Passengers', scatter=False, color='red')
        plt.show()

    def airports(self):
        """
        Top and bottom airports by passenger count.

        Returns
        -------
        None
        """
        # Combine Origin and Destination columns to create an 'Airport' column
        self.df['Airport'] = self.df['Origin']
        top_airports = self.df.groupby('Airport')['Passengers'].sum().sort_values(ascending=False).head(10)
        bottom_airports = self.df.groupby('Airport')['Passengers'].sum().sort_values().head(10)
        print("Top Airports:")
        print(top_airports)
        print("\nBottom Airports:")
        print(bottom_airports)
        
    def state_passenger_data(self):
        """
        Top states by passenger count.

        Returns
        -------
        None
        """
        state_passenger_data = self.df.groupby('Origin_State')['Passengers'].sum().sort_values(ascending=False)
        print("Top States by Passenger Count:")
        print(state_passenger_data)
        
    def seat_occupancy_year(self):
        """
        Seat occupancy trends over the years.

        Returns
        -------
        None
        """
        self.df['Seat_Occupancy'] = self.df['Passengers'] / self.df['Seats']
        seat_occupancy_over_time = self.df.groupby('Year')['Seat_Occupancy'].mean()
        seat_occupancy_over_time.plot(kind='line')
        plt.title('Seat Occupancy Trends Over the Years')
        plt.xlabel('Year')
        plt.xticks(range(1990, 2010, 2))
        plt.ylabel('Average Seat Occupancy')
        plt.show()        

    def seat_occupancy_month(self):
        """
        Seat occupancy by month.

        Returns
        -------
        None
        """
        seat_occupancy_by_month = self.df.groupby('Month')['Seat_Occupancy'].mean()
        seat_occupancy_by_month.plot(kind='bar')
        plt.title('Seat Occupancy by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Seat Occupancy')
        plt.show()        
        
    def route_seat_occupancy(self):
        """
        Average seat occupancy by route.

        Returns
        -------
        None
        """
        route_seat_occupancy = self.df.groupby('Route')['Seat_Occupancy'].mean().sort_values(ascending=False)
        print("Average Seat Occupancy by Route:")
        print(route_seat_occupancy)
        
    def Seat_Occupancy(self):
        """
        Sort rows by seat occupancy.

        Returns
        -------
        None
        """
        print(self.df.Seat_Occupancy.sort_values(ascending=False))
        
    def Examine_Seat_Occupancy(self):
        """
        Examine seat occupancy of a specific row and find rows with zero seats.

        Returns
        -------
        None
        """
        row_2801180 = self.df.loc[2801180]
        print("Details of Row 2801180:")
        print(row_2801180) 

        num_rows_with_zero_seats = len(self.df[self.df['Seats'] == 0])
        print("Number of Rows with Zero Seats:", num_rows_with_zero_seats)
        
    def Modify_Seat_Occupancy(self):
        """
        Replace zero seats with corresponding passengers in the DataFrame.

        Returns
        -------
        None
        """
        # Replace zero Seats with corresponding Passengers
        self.df['Seats'] = np.where(self.df['Seats'] == 0, self.df['Passengers'], self.df['Seats'])
        row_2801180 = self.df.loc[2801180]
        print("Details of Row 2801180 after Modification:")
        print(row_2801180)
        
    def Seat_Occupancy_new(self):
        """
        Calculate and display seat occupancy as a new column.

        Returns
        -------
        None
        """
        self.df['Seat_Occupancy_new'] = self.df['Passengers'] / self.df['Seats']
        print(self.df.Seat_Occupancy_new.sort_values(ascending=False))
        
    def Seat_Occupancy_new_year(self):
        """
        Seat occupancy trends with the new column over the years.

        Returns
        -------
        None
        """
        seat_occupancy_over_time = self.df.groupby('Year')['Seat_Occupancy_new'].mean()
        seat_occupancy_over_time.plot(kind='line')
        plt.title('Seat Occupancy Trends Over the Years (New Column)')
        plt.xlabel('Year')
        plt.xticks(range(1990, 2010, 2))
        plt.ylabel('Average Seat Occupancy')
        plt.show()

    def Seat_Occupancy_new_month(self):
        """
        Seat occupancy by month using the new column.

        Returns
        -------
        None
        """
        seat_occupancy_by_month = self.df.groupby('Month')['Seat_Occupancy_new'].mean()
        seat_occupancy_by_month.plot(kind='bar')
        plt.title('Seat Occupancy by Month (New Column)')
        plt.xlabel('Month')
        plt.ylabel('Average Seat Occupancy')
        plt.show()

    def passengers_distance_matplotlib(self):
        """
        Create a scatter plot of seat occupancy vs. flight distance using Matplotlib.

        Returns
        -------
        None
        """
        plt.scatter(self.df['Distance'], self.df['Seat_Occupancy_new'], alpha=0.5)
        plt.title('Seat Occupancy vs. Flight Distance')
        plt.xlabel('Distance (Miles)')
        plt.ylabel('Seat Occupancy')
        plt.ylim(0, 2)  # Set the y-axis range to be between 0 and 2
        plt.show()

    def passengers_distance_seaborn(self):
        """
        Create a scatter plot of seat occupancy vs. flight distance using Seaborn.

        Returns
        -------
        None
        """
        # Set a professional style for the plot
        sns.set(style='whitegrid', context='notebook', palette='dark')

        # Create the scatter plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        sns.scatterplot(data=self.df, x='Distance', y='Seat_Occupancy_new', alpha=0.5)

        # Set labels and title
        plt.xlabel('Distance (Miles)')
        plt.ylabel('Seat Occupancy')
        plt.title('Seat Occupancy vs. Flight Distance')

        # Set the y-axis range to be between 0 and 2
        plt.ylim(0, 2)

        plt.show()
        