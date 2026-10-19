
#!/usr/bin/env python3
"""Module for removing the customerID column."""


def drop_customerID(df):
    """Drop the customerID column from the DataFrame."""
    return df.drop(columns=['customerID'])

