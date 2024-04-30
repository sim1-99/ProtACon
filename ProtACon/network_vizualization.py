'''
This script vizualize the network in a 3D plota.

'''
import plotly.graph_objs as go
import igraph as ig
from ProtACon.modules import miscellaneous as misc

# HACK organize the main to receive bool flags for the properties in order to be reuseable in the cla later

def main(
        AA_features : pd.DataFrame, 
        link_property : list[edge : tuple]
        ) -> go.Figure:
    pass

def plot_network_3D():

    pass

def enhanche_property1()-> list[tuple]: # return the list of edges to be colored
    pass

def enhanche_property2():
    pass

def enhanche_property3():
    pass

def enhanche_property4():
    pass


def plot3D_network_1st_method():
    
    data = [] # upload the dataframe here

    # Put N as the n° of AA
    N = len(data['AA_idx'])

    # Crea la lista degli archi e l'oggetto Graph da questi archi
    L = len(data['links'])
    Edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
    G = ig.Graph(Edges, directed=False)

    # Estrai gli attributi dei nodi, 'group' e 'name'
    labels = [node['name'] for node in data['nodes']]
    group = [node['group'] for node in data['nodes']]

    # Ottieni le posizioni dei nodi utilizzando il layout Kamada-Kawai per i grafici 3D
    layt = G.layout('kk', dim=3)

    # Estrai le coordinate dei nodi
    Xn = [layt[k][0] for k in range(N)]
    Yn = [layt[k][1] for k in range(N)]
    Zn = [layt[k][2] for k in range(N)]

    # Estrai le coordinate degli archi
    Xe = []
    Ye = []
    Ze = []
    for e in Edges:
        Xe += [layt[e[0]][0], layt[e[1]][0], None]
        Ye += [layt[e[0]][1], layt[e[1]][1], None]
        Ze += [layt[e[0]][2], layt[e[1]][2], None]

    # Crea il tracciato per gli archi
    trace1 = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=1),
        hoverinfo='none'
    )

    # Crea il tracciato per i nodi
    trace2 = go.Scatter3d(
        x=Xn,
        y=Yn,
        z=Zn,
        mode='markers',
        name='attori',
        marker=dict(
            symbol='circle',
            size=6,
            color=group,
            colorscale='Viridis',
            line=dict(color='rgb(50,50,50)', width=0.5)
        ),
        text=labels,
        hoverinfo='text'
    )

    # Imposta il layout del grafico
    layout = go.Layout(
        title="Rete delle coapparizioni dei personaggi nel romanzo di Victor Hugo<br>Les Misérables (visualizzazione 3D)",
        width=1000,
        height=1000,
        showlegend=False,
        scene=dict(
            xaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')
        ),
        margin=dict(t=100),
        hovermode='closest',
        annotations=[
            dict(
                showarrow=False,
                text="Fonte dati: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>miserables.json</a>",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=14)
            )
        ],
    )

    # Crea la figura
    fig = go.Figure(data=[trace1, trace2], layout=layout)

    # Mostra il grafico
fig.show()
pass
# HACK use the method g=ig.Graph() g.add_edges() to get the edges list from a binarized map
def plot3D_network_2nd_method():
    import plotly.graph_objects as go

# Definisci il dizionario con le coordinate
coordinates = {
    'A': (4.2, 8.7, 3.1),
    'B': (9.8, 2.3, 6.5),
    'C': (7.1, 5.4, 9.2),
    'D': (2.9, 6.8, 1.5),
    'E': (5.6, 3.2, 7.8)
}

# Estrai le coordinate x, y, z
x_coords, y_coords, z_coords = zip(*coordinates.values())

# Definisci i nodi del network
nodes = list(coordinates.keys())

# Definisci gli archi del network
edges = [(0, 1, '0&1'), (1, 2, '1&2'), (2, 3, '2&3'), (3, 4, '3&4')]

# Crea il grafico del network
fig = go.Figure()

# Aggiungi i nodi al grafico
for node, x, y, z in zip(nodes, x_coords, y_coords, z_coords):
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode="markers+text", text=[node], textposition="top center"))

# Aggiungi gli archi al grafico
for edge in edges:
    x_edge = [x_coords[edge[0]], x_coords[edge[1]]]
    y_edge = [y_coords[edge[0]], y_coords[edge[1]]]
    z_edge = [z_coords[edge[0]], z_coords[edge[1]]]
    fig.add_trace(go.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode="lines+text", text=edge[2], line=dict(width=2)))

# Imposta il layout
fig.update_layout(scene=dict(aspectmode="cube"))

# Mostra il grafico
fig.show()


    pass