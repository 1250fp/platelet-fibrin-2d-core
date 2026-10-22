"""Named, inspectable mechanics profiles for Program19.3.

Profiles are intentionally orthogonal.  Selecting a literal CodeOcean
equation does not silently select its XML values, fixed-step integrator, or
3-D geometry.  ``make_profiled_config`` is the only function that applies
presets; ordinary JSON loading preserves every recorded value verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Program19Config


@dataclass(frozen=True)
class ProfileDescription:
    title: str
    purpose: str
    main_differences: tuple[str, ...]


PROFILE_DESCRIPTIONS: dict[str, dict[str, ProfileDescription]] = {
    "equation": {
        "paper_consistent": ProfileDescription(
            "Paper-consistent corrected equations",
            "Recommended for revised minipaper and thesis results.",
            (
                "Analytical WLC tangent with the cubic denominator.",
                "Rest-angle bending energy with an explicit dual length.",
                "Dimensionally labelled platelet pair potential.",
            ),
        ),
        "codeocean_literal": ProfileDescription(
            "Literal CodeOcean equation conventions",
            "Source-comparison and sensitivity only.",
            (
                "Absolute mean local strain and first-power stiffness denominator.",
                "Planar counterpart of the CUDA normalized-direction torsion force.",
                "Integer-division and double-force-factor convention in platelet LJ.",
            ),
        ),
    },
    "parameter": {
        "thesis_audited": ProfileDescription(
            "Audited thesis baseline",
            "Conservative continuation of the audited Program17/18/19 line.",
            (
                "47.6 um domain (48 um for generated density_table clots) and f0=2.233 nN.",
                "Michael WLC constants, but explicitly separated platelet forces.",
                "Restricted cohesion valence with fiber identity.",
            ),
        ),
        "minipaper_revision": ProfileDescription(
            "Revised minipaper baseline",
            "Recommended source for new f0, porosity, MET, and R_agg figures.",
            (
                "60 um domain and f0=2.2 nN baseline matching the proposed sweep.",
                "Uses audited mechanics instead of copying inconsistent draft notation.",
                "Keeps adhesion force distinct from filopodial baseline f0.",
            ),
        ),
        "codeocean_data_xml": ProfileDescription(
            "CodeOcean README data.xml values",
            "Comparison with the public README command: bend_model -dt=0.0001 data.xml.",
            (
                "N=50, platelet force=6.1, and exactly 3 filopodia per platelet.",
                "Filopodial force is divided by the number of attached arms.",
                "Cohesion has no fiber-identity exclusion and permits high node degree.",
            ),
        ),
        "codeocean_data_pdf": ProfileDescription(
            "CodeOcean data_pdf.xml values",
            "Comparison with the alternate public XML, not the README default.",
            (
                "Platelet force=2.0 and up to 10 lognormally sampled filopodia.",
                "Lognormal parameters M=2 and S=0.75 use rejection and truncation.",
                "Filopodial force is divided by attached-arm count.",
            ),
        ),
        "custom": ProfileDescription(
            "Custom recorded parameters",
            "For calibrated or exploratory runs that do not claim a named preset.",
            ("No preset values are imposed.",),
        ),
    },
    "numerical": {
        "corrected_adaptive": ProfileDescription(
            "Corrected adaptive overdamped integration",
            "Recommended for scientific results.",
            (
                "Forces are recomputed after adaptive substeps.",
                "Independent Gaussian thermal increments are used in each coordinate.",
                "No force or displacement clipping; unsafe requests abort.",
            ),
        ),
        "codeocean_fixed_step": ProfileDescription(
            "CodeOcean-like fixed-step integration",
            "Compatibility diagnosis, not the recommended diffusion model.",
            (
                "Fixed Euler step with no adaptive substeps.",
                "Uniform random increment shared by both projected coordinates.",
                "No center-of-mass translation removal.",
            ),
        ),
    },
    "geometry": {
        "generated_2d": ProfileDescription(
            "Generated planar clot",
            "Independent 2-D realization used by the thesis model.",
            (
                "Degree-3/4 planar graph generated from the configured seed.",
                "Explicit fiber identities and adaptive edge subdivision.",
            ),
        ),
        "codeocean_xml_xy": ProfileDescription(
            "CodeOcean XML projected onto XY",
            "Force-by-force comparison from the same explicit public network.",
            (
                "Reads XML nodes, links, platelets, and fixed-node identifiers.",
                "Centers the XY projection without inventing z-layers.",
                "Projection is not equivalent to the original 3-D torsion geometry.",
            ),
        ),
    },
}


PARAMETER_PRESETS: dict[str, dict[str, Any]] = {
    "thesis_audited": {},
    "minipaper_revision": {
        "domain_size_um": 60.0,
        "filopodia_f0_nN": 2.2,
        "platelet_fibrin_force_nN": 4.0,
        "platelet_platelet_force_parameter": 4.0,
    },
    "codeocean_data_xml": {
        "monomers_in_parallel_N": 50.0,
        "persistence_length_um": 0.001074033502990,
        "contour_multiplier_C": 2.351243606987286,
        "bending_EI_nN_um2": 0.002262593420685,
        "archive_bending_force_nN": 0.002262593420685,
        "segment_target_um": 0.30,
        "cohesion_distance_um": 0.10,
        "cohesion_search_stride": 1,
        "cohesion_max_new_per_node_per_search": 20,
        "cohesion_max_total_neighbors": 100,
        "cohesion_exclude_same_fiber": False,
        "cohesion_degree_includes_backbone": True,
        "platelet_radius_um": 1.13,
        "platelet_core_fraction": 0.90,
        "platelet_force_radius_um": 2.18,
        "platelet_fibrin_force_nN": 6.1,
        "platelet_platelet_force_parameter": 6.1,
        "filopodia_count_model": "fixed",
        "filopodia_fixed_count": 3,
        "filopodia_min": 3,
        "filopodia_max": 3,
        "filopodia_selection_rule": "codeocean_random",
        "filopodia_force_distribution": "divide_by_attached",
        "filopodia_global_node_exclusivity": False,
        "filopodia_handover_rule": "graph_neighbor",
        "filopodia_f0_nN": 2.233,
    },
    "codeocean_data_pdf": {
        "monomers_in_parallel_N": 50.0,
        "persistence_length_um": 0.001074033502990,
        "contour_multiplier_C": 2.351243606987286,
        "bending_EI_nN_um2": 0.002262593420685,
        "archive_bending_force_nN": 0.002262593420685,
        "segment_target_um": 0.30,
        "cohesion_distance_um": 0.10,
        "cohesion_search_stride": 1,
        "cohesion_max_new_per_node_per_search": 20,
        "cohesion_max_total_neighbors": 100,
        "cohesion_exclude_same_fiber": False,
        "cohesion_degree_includes_backbone": True,
        "platelet_radius_um": 1.13,
        "platelet_core_fraction": 0.90,
        "platelet_force_radius_um": 2.18,
        "platelet_fibrin_force_nN": 2.0,
        "platelet_platelet_force_parameter": 2.0,
        "filopodia_count_model": "codeocean_lognormal_rejection",
        "filopodia_lognormal_mu": 2.0,
        "filopodia_lognormal_sigma": 0.75,
        "filopodia_min": 1,
        "filopodia_max": 10,
        "filopodia_selection_rule": "codeocean_random",
        "filopodia_force_distribution": "divide_by_attached",
        "filopodia_global_node_exclusivity": False,
        "filopodia_handover_rule": "graph_neighbor",
        "filopodia_f0_nN": 2.233,
    },
    "custom": {},
}


NUMERICAL_PRESETS: dict[str, dict[str, Any]] = {
    "corrected_adaptive": {
        "adaptive_substeps": True,
        "thermal_noise_model": "gaussian_independent",
        "remove_global_translation": True,
    },
    "codeocean_fixed_step": {
        "dt_s": 0.0001,
        "adaptive_substeps": False,
        "thermal_noise_model": "codeocean_uniform_shared",
        "remove_global_translation": False,
        "mobility_model": "effective_network",
    },
}


def make_profiled_config(
    equation_profile: str = "paper_consistent",
    parameter_profile: str = "thesis_audited",
    numerical_profile: str = "corrected_adaptive",
    geometry_profile: str = "generated_2d",
    overrides: dict[str, Any] | None = None,
) -> Program19Config:
    """Create a fully resolved config; explicit overrides have final precedence."""
    values = Program19Config().to_dict()
    # Leave the domain unresolved so its default follows the final
    # node_degree_model / geometry_profile (see Program19Config.__post_init__).
    values["domain_size_um"] = None
    values.update(PARAMETER_PRESETS[parameter_profile])
    values.update(NUMERICAL_PRESETS[numerical_profile])
    values.update(
        equation_profile=equation_profile,
        parameter_profile=parameter_profile,
        numerical_profile=numerical_profile,
        geometry_profile=geometry_profile,
    )
    if geometry_profile == "codeocean_xml_xy":
        values["remove_global_translation"] = False
    if overrides:
        values.update(overrides)
    return Program19Config.from_dict(values)


def profile_audit(config: Program19Config) -> dict[str, Any]:
    """Return descriptions and deviations from the named parameter presets."""
    expected = {}
    expected.update(PARAMETER_PRESETS.get(config.parameter_profile, {}))
    expected.update(NUMERICAL_PRESETS.get(config.numerical_profile, {}))
    actual = config.to_dict()
    deviations = {
        key: {"profile_value": value, "run_value": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    selected = {
        "equation": config.equation_profile,
        "parameter": config.parameter_profile,
        "numerical": config.numerical_profile,
        "geometry": config.geometry_profile,
    }
    return {
        "selected": selected,
        "descriptions": {
            axis: {
                "title": PROFILE_DESCRIPTIONS[axis][name].title,
                "purpose": PROFILE_DESCRIPTIONS[axis][name].purpose,
                "main_differences": list(
                    PROFILE_DESCRIPTIONS[axis][name].main_differences
                ),
            }
            for axis, name in selected.items()
        },
        "deviations_from_named_presets": deviations,
        "interpretation": (
            "Deviations are allowed and recorded. Use parameter_profile=custom when "
            "the deviations define a new calibrated parameter set."
        ),
    }


def profiles_markdown() -> str:
    lines = ["# Program19.3 profile guide", ""]
    for axis, entries in PROFILE_DESCRIPTIONS.items():
        lines.extend((f"## {axis.title()} profiles", ""))
        for name, description in entries.items():
            lines.extend(
                (
                    f"### `{name}` - {description.title}",
                    "",
                    description.purpose,
                    "",
                )
            )
            lines.extend(f"- {item}" for item in description.main_differences)
            lines.append("")
    return "\n".join(lines)
