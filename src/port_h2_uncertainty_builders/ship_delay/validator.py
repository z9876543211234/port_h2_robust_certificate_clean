"""Independent validation entry point for ship-delay Bundles."""

from port_h2_contracts.uncertainty_bundle import UncertaintyBundle


def validate_ship_delay_bundle(bundle: UncertaintyBundle) -> None:
    if bundle.bundle_type != "ship_delay":
        raise ValueError("expected a ship_delay UncertaintyBundle")
    bundle.validate()

