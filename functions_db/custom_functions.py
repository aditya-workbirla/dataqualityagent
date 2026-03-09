
# verified_by_team = False
def check_negative_pressure_values(series):
    negative_values = series[series < 0]
    return {
        'negative_values_count': len(negative_values),
        'negative_values_percentage': len(negative_values) / len(series) * 100,
        'negative_values': negative_values.tolist()
    }



# verified_by_team = False
def check_all_negative_values(series):
    negative_values = series[series < 0]
    all_negative = len(negative_values) == len(series)
    return {'all_values_negative': all_negative, 'negative_count': len(negative_values)}
