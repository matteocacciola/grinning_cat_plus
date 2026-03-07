from cat import hook, CheshireCat, BillTheLizard


@hook(priority=0)
def after_cheshire_cat_creation(cat: CheshireCat, lizard: BillTheLizard) -> None:
    this_plugin_id = lizard.mad_hatter.get_plugin().id
    cat.plugin_manager.toggle_plugin(this_plugin_id)
