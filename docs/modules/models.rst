:mod:`earth2studio.models`: Models
----------------------------------
.. automodule:: earth2studio.models
    :no-members:
    :no-inherited-members:

.. currentmodule:: earth2studio

.. _earth2studio.models.px:

:mod:`earth2studio.models.px`: Prognostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Prognostic models are a class of models that perform time-integration.
Thus are typically used to generate forecast predictions.

.. warning ::

   Pre-trained prognostic models provided in Earth2Studio may be provided
   under different licenses. We encourage users to familiarize themselves with each
   prior to use.

.. autosummary::
   :nosignatures:
   :toctree: generated/models/px/
   :template: prognostic.rst

   models.px.AIFS
   models.px.Aurora
   models.px.DLESyM
   models.px.DLESyMLatLon
   models.px.DLWP
   models.px.FCN
   models.px.FengWu
   models.px.FuXi
   models.px.GraphCastSmall
   models.px.InterpModAFNO
   models.px.Pangu24
   models.px.Pangu6
   models.px.Pangu3
   models.px.Persistence
   models.px.SFNO
   models.px.StormCast

.. _earth2studio.models.dx:

:mod:`earth2studio.models.dx`: Diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Diagnostic models are a class of models that do not perform time-integration.
These may be used to map between weather/climate variables to other quantities of
interest, used to enbable additional analysis, improve prediction accuracy, downscale,
etc.

.. warning ::

   Pre-trained diagnostic models provided in Earth2Studio may be provided
   under different licenses. We encourage users to familiarize themselves with each
   prior to use.

.. autosummary::
   :nosignatures:
   :toctree: generated/models/dx/
   :template: diagnostic.rst

   models.dx.CorrDiffTaiwan
   models.dx.ClimateNet
   models.dx.DerivedRH
   models.dx.DerivedRHDewpoint
   models.dx.DerivedVPD
   models.dx.DerivedWS
   models.dx.PrecipitationAFNO
   models.dx.PrecipitationAFNOv2
   models.dx.SolarRadiationAFNO1H
   models.dx.SolarRadiationAFNO6H
   models.dx.TCTrackerWuDuan
   models.dx.TCTrackerVitart
   models.dx.WindgustAFNO
   models.dx.Identity

:mod:`earth2studio.models`: Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :nosignatures:
   :toctree: generated/models/
   :template: class.rst

   models.auto.Package
   models.batch.batch_func
