# Makefile for Learn2Feel
PACKAGE_NAME:=Learn2Feel

# Some helpers
define print_clean
	echo "\033[0;36m[$(PACKAGE_NAME)] Cleaning $(1) package\033[0m"
endef

define print_install_pack
	echo "\033[1;32m[$(PACKAGE_NAME)] Installing $(1)...\033[0m"
endef

define pip_install
	cd $(1) && $(if $(2),pip install $(2),pip install -U --no-cache-dir .)
endef

define pip_install_develop
	cd $(1) && $(if $(2),pip install $(2),pip install -U --no-cache-dir -e .)
endef

define pip_uninstall
	yes Y | pip uninstall $(1)
endef

define data_download
	mkdir -p data
	wget -O data/learn2feel_data.hdf5 https://edmond.mpdl.mpg.de/api/access/datafile/211961
endef

# Targets
.PHONY: all learn2feel clean clean_learn2feel info

all: learn2feel learn2feel_data

learn2feel:
	@$(call print_install_pack,$@)
	@pip install .

learn2feel_data:
	@echo "Downloading data to relative location /data/learn2feel_data.hdf5."
	@$(call data_download) 

# Info
info:
	@echo "\033[1;32m[$(PACKAGE_NAME)] Using interpreter `which python` (version `python --version | cut -d' ' -f2`)\033[0m"
	@echo "Installed packages: "
	@pip list

# Clean targets
clean: clean_learn2feel

clean_learn2feel:
	@$(call print_clean,learn2feel)
	@$(call pip_uninstall,learn2feel)