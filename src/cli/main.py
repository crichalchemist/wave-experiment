"""Command-line interface for Detective LLM."""

import click


@click.group()
def cli():
    """Detective LLM: Information Gap Analysis System"""
    pass


@cli.command()
@click.argument('claim')
def analyze(claim: str):
    """Analyze a claim for information gaps."""
    click.echo(f"Analyzing: {claim}")
    # TODO: Implement analysis
    

@cli.command()
@click.option('--entity', required=True)
@click.option('--hops', default=2)
def network(entity: str, hops: int):
    """Trace network connections from an entity."""
    click.echo(f"Tracing network from {entity} ({hops} hops)")
    # TODO: Implement network tracing


if __name__ == '__main__':
    cli()
