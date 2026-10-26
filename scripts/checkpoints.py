"""Fixed Part A reference values transcribed from the assignment, not outputs."""

RAW_METRICS = {
    "raw_orders": 13084, "exact_duplicates": 119, "pending_rows": 656,
    "distinct_order_ids": 12309, "non_numeric_amounts": 120,
    "negative_delivery_times": 106, "blank_delivery_times": 2541,
    "blank_ages": 61, "invalid_ages": 6, "raw_city_spellings": 24,
    "nonblank_gender_spellings": 10, "blank_genders": 30,
}
CITY_COUNTS = {
    "Pune": 227, "Delhi": 210, "Bengaluru": 197,
    "Chennai": 191, "Hyderabad": 191, "Mumbai": 184,
}
GENDER_COUNTS = {"female": 610, "male": 560, "unknown": 30}
REVENUE = 2998668.80
SQL_ROW_COUNTS = {"Q1": 30, "Q2": 30, "Q3": 11, "Q4": 25, "Q5": 550, "Q6": 1175, "Q7": 183}
Q6_LEADERS = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10),
}
CITY_AGGREGATES = {
    "Bengaluru": (523643.40, 276.47, 1894),
    "Mumbai": (479058.50, 297.00, 1613),
}
