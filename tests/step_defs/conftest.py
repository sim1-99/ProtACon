"""
Copyright (c) 2024 Simone Chiarella

Author: S. Chiarella
Date: 2024-11-18

Fixtures for testing both the general functioning and the behavior in specific
situations, of the step definitions of the BDD features.

"""
import pytest
import responses


@pytest.fixture
def url(chain_ID):
    """URL to the PDB file relative to the PDB chain ID."""
    return f"https://files.rcsb.org/download/{chain_ID}.pdb"


@pytest.fixture
def resp_headers(chain_ID):
    """Headers to pass to mock_rcsb_response."""
    return {"Content-Disposition": f"filename={chain_ID.lower()}.pdb"}


@pytest.fixture
def mock_rcsb_response(resp_headers, url):
    """
    Mocked response from the RCSB PDB URL, that simulates the real content of
    the header field "Content-Disposition".

    """
    with responses.RequestsMock() as mock:
        mock.get(
            url=url,
            headers=resp_headers,
            status=200,
        )
        yield mock


@pytest.fixture
def response_url(mock_rcsb_response):
    """The URL called in the response."""
    return mock_rcsb_response.calls[0].response.url


@pytest.fixture
def response_call_count(mock_rcsb_response):
    """The number of times the response was called."""
    return len(mock_rcsb_response.calls)


@pytest.fixture
def response_status_code(mock_rcsb_response):
    """The status code of the response."""
    return mock_rcsb_response.calls[0].response.status_code
