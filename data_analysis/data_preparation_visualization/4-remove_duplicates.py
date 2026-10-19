#!/usr/bin/env python3
"""Module for removing duplicate rows from a DataFrame."""


def remove_duplicates(df):
    """Remove duplicate rows and return the cleaned DataFrame."""
    return df.drop_duplicates()
