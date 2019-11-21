from ._base import _Model1D, _ModelHD
from ._typing import TimeSeries, Series, DataFrame, Any


class _Transformer1D(_Model1D):
    def fit(self, ts: TimeSeries) -> None:
        """Train the transformer with given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used to train the transformer.
            If a DataFrame with k columns, k univariate transformers will be
            trained independently.

        """
        self._fit(ts)

    def transform(self, ts: TimeSeries) -> TimeSeries:
        """Transform time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be transformed.
            If a DataFrame with k columns, k univariate transformers will be
            applied to them independently.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        return self._predict(ts)

    def fit_transform(self, ts: TimeSeries) -> TimeSeries:
        """Train the transformer, and tranform the time series used for
        training.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used for training and be transformed.
            If a DataFrame with k columns, k univariate transformers will be
            applied to them independently.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        self.fit(ts)
        return self.predict(ts)

    def predict(self, ts: TimeSeries, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Alias of `transform`.
        """
        return self.transform(ts)

    def fit_predict(self, ts: TimeSeries, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Alias of `fit_transform`.
        """
        return self.fit_transform(ts)


class _TransformerHD(_ModelHD):
    def fit(self, df: DataFrame) -> None:
        """Train the transformer with given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used to train the transformer.

        """
        self._fit(df)

    def transform(self, df: TimeSeries) -> TimeSeries:
        """Transform time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be transformed.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        return self._predict(df)

    def fit_transform(self, df: TimeSeries) -> TimeSeries:
        """Train the transformer, and tranform the time series used for
        training.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used for training and be transformed.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        self.fit(df)
        return self.predict(df)

    def predict(self, df: TimeSeries, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Alias of `transform`.
        """
        return self.transform(df)

    def fit_predict(self, df: TimeSeries, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Alias of `fit_transform`.
        """
        return self.fit_transform(df)
