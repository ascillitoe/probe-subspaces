import numpy as np

def polar_sort(points):
    N = points.shape[0]
    
    # compute centroid
    cent=(sum([points[i,0] for i in range(N)])/N,sum([points[i,1] for i in range(N)])/N)
    # sort by increasing polar angle
    polar = np.arctan2(points[:,1]-cent[1],points[:,0]-cent[0])
    index = np.argsort(polar)
    return points[index]

def grid_within_hull(points,ax=None,N=20):
    from scipy.spatial import ConvexHull

    # Create grid within convex hull (i.e. perimeter of our current u2 points)
    hull = ConvexHull(points)

    if ax is not None:
        for simplex in hull.simplices:
            ax.plot(points[simplex,0], points[simplex,1], 'C3--',lw=2)
    
    verts = list(zip(points[hull.vertices,0],points[hull.vertices,1]))
    xx, yy = np.meshgrid(np.linspace(-2,2,N), np.linspace(-2,2,N),indexing='ij')
    x, y = xx.flatten(), yy.flatten()
    points = np.vstack((x,y)).T
    from matplotlib.path import Path
    path = Path(verts)
    grid = path.contains_points(points)
    grid = grid.reshape((N,N))
    
    if ax is not None:
        for i in range(N):
            y_in_hull = yy[i,:][grid[i,:]]
            if(len(y_in_hull)>1):
                ymin = np.min(y_in_hull)
                ymax = np.max(y_in_hull)
                ax.vlines(xx[i,0],ymin,ymax,lw=2)
        
        for j in range(N):
            x_in_hull = xx[:,j][grid[:,j]]
            if(len(x_in_hull)>1):
                xmin = np.min(x_in_hull)
                xmax = np.max(x_in_hull)
                ax.hlines(yy[0,j],xmin,xmax,lw=2)

    return list(zip(xx[grid],yy[grid]))

def mean_std_X(subspace,coord,ND=1000):
    coord = np.array(coord).T
    X = subspace.get_samples_constraining_active_coordinates(ND,coord)
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    return mean,std


###################

def make_colormap(seq):
    import matplotlib.colors as mcolors

    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    #%
    cdict = {'red': [], 'green': [], 'blue': []}

    # make a lin_space with the number of records from seq.     
    x = np.linspace(0,1, len(seq))
    #%
    for i in range(len(seq)):
        segment = x[i]
        tone = seq[i]
        cdict['red'].append([segment, tone, tone])
        cdict['green'].append([segment, tone, tone])
        cdict['blue'].append([segment, tone, tone])
    #%
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def standardise(X,min_value=None,max_value=None):
    M,d=X.shape
    X_stnd=np.zeros((M,d))
    if min_value is None:
        min_value = np.empty(d)
        max_value = np.empty(d)
        for j in range(0,d):
            max_value[j] = np.max(X[:,j])
            min_value[j] = np.min(X[:,j])
    for j in range(0,d):
        for i in range(0,M):
            X_stnd[i,j]=2.0 * ( (X[i,j]-min_value[j])/(max_value[j] - min_value[j]) ) -1
    return X_stnd, min_value, max_value

