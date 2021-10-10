# Pluto-SDR: Passive Radar

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/ChiefGokhlayeh/pluto-sdr-pr/main.svg)](https://results.pre-commit.ci/latest/github/ChiefGokhlayeh/pluto-sdr-pr/main)
[![pytest](https://github.com/ChiefGokhlayeh/pluto-sdr-pr/actions/workflows/pytest.yml/badge.svg)](https://github.com/ChiefGokhlayeh/pluto-sdr-pr/actions/workflows/pytest.yml)
[![doc-build](https://github.com/ChiefGokhlayeh/pluto-sdr-pr/actions/workflows/doc-build.yml/badge.svg)](https://github.com/ChiefGokhlayeh/pluto-sdr-pr/actions/workflows/doc-build.yml)
[![doc-lint](https://github.com/ChiefGokhlayeh/pluto-sdr-pr/actions/workflows/doc-lint.yml/badge.svg)](https://github.com/ChiefGokhlayeh/pluto-sdr-pr/actions/workflows/doc-lint.yml)

Student project aiming to implement [passive radar](https://en.wikipedia.org/wiki/Passive_radar) functionality, using cheap ~~RTL-SDR receivers~~ **Pluto-SDR receivers** and public **LTE-based 5G Broadcast** radio signals as illuminators of opportunity.

This project has an explorative character. The goal is to learn about signal processing and radio technologies.

## Development

Use the provided devcontainer image to create a ready-to-use NumPy environment. In vscode download the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension and re-open this workspace inside said development image.

Most of the prototyping work happens in Jupyter notebooks. Once cured, code is refactored into reusable Python modules.

## Contributors

Main contributors on this project are (sorted by lastname):

-   Andreas Baulig
-   Wolfgang Bradfisch

## Switch from RTL-SDR to Pluto-SDR

The project started out looking into the feasibility of using cheap DVB-T2 receivers, aka [RTL-SDR](https://www.rtl-sdr.com/), for passive radar in combination with LTE-based 5G Broadcast illuminators. Unfortunately, the bandwidth requirements associated with typical LTE-based 5G Broadcast stations are beyond the capabilities of these receivers. Now wait, aren't these receivers meant for DVB-T2 reception? Correct, and in normal operational use they are capable of up to 10 MHz channel bandwidth, as required by DVB-T2 standards. However, to use them as SDR receivers we have to reconfigure their firmware to operate in a sparsely documented _debug mode_ (at least most official documentation for the RTL2832U chipset protected under NDA). In this mode the bandwidth is limited to an unstable 3.2 MHz (or 2.56 MHz stable), too low for the LTE channel we're investigating at 5 MHz.

That's why we've **switched hardware** and are now using **two coherent [Pluto-SDR](https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/adalm-pluto.html) receivers** fed by an external [**GPS-DO**](http://www.leobodnar.com/shop/index.php?main_page=product_info&cPath=107&products_id=301&zenid=c7c15007aa38f805496e24675ece8b70). Not as cheap as initially planned, but still within a moderate hobbyist budget.

Maybe in the future one could experiment with lowering the available bandwidth to 2.56 MHz. That would still be enough to decorrelate the LTE synchronization signals, as PSS and SSS are located in the center 5 resource blocks equating to 1 MHz minimum bandwidth. A lot of energy would be lost left and right of the receiver channel, but perhaps it could still be usable as a toy project.
