"""Paired seasonal OLS and actual-year-lag Bartlett inference.

The supplied predictor retains its original scale. A single optional control
and a linear actual-year trend are fitted on the same finite years per cell.
"""
import numpy as np
from scipy import stats


def detrend(values, years, valid):
    """One OLS line over every valid actual year; centered residuals."""
    n = valid.sum(axis=0)
    t = (years-years.mean())[:,None]
    mean_t = np.divide((t*valid).sum(0), n, out=np.zeros(n.shape,float), where=n>0)
    mean_y = np.divide(np.where(valid,values,0).sum(0), n, out=np.zeros(n.shape,float), where=n>0)
    tc = np.where(valid,t-mean_t,0)
    yc = np.where(valid,values-mean_y,0)
    denominator = (tc*tc).sum(0)
    slope = np.divide((tc*yc).sum(0), denominator, out=np.zeros(n.shape,float), where=denominator>0)
    return np.where(valid,yc-tc*slope,np.nan), slope


def pearson(x, y, valid):
    n=valid.sum(0)
    mx=np.divide(np.where(valid,x,0).sum(0),n,out=np.zeros(n.shape,float),where=n>0)
    my=np.divide(np.where(valid,y,0).sum(0),n,out=np.zeros(n.shape,float),where=n>0)
    xx=np.where(valid,x-mx,0); yy=np.where(valid,y-my,0)
    denominator=np.sqrt((xx*xx).sum(0)*(yy*yy).sum(0))
    r=np.divide((xx*yy).sum(0),denominator,out=np.full(n.shape,np.nan),where=(n>=3)&(denominator>0))
    return np.clip(r,-1,1),n


def pvalue(r, dof):
    good=np.isfinite(r)&(dof>0)
    t=np.zeros(r.shape,float)
    t[good]=np.abs(r[good])*np.sqrt(dof[good]/np.maximum(1-r[good]**2,1e-15))
    return np.where(good,2*stats.t.sf(t,dof),np.nan)


def correlate(predictor, field, years, min_seasons=10, max_lag=3, control=None):
    """Fit both trends on the full paired record at each grid cell.

    Paired masking avoids using different time samples for the two trend fits.
    Actual year-distance pairs are used for Bartlett lags, including gaps.
    """
    shape=field.shape[1:]
    y=np.asarray(field,float).reshape(len(years),-1)
    x=np.broadcast_to(np.asarray(predictor,float)[:,None],y.shape)
    years=np.asarray(years,float)
    if len(np.unique(years))!=len(years) or not np.all(years==np.rint(years)):
        raise ValueError('Season years must be unique integer model/calendar years')
    valid=np.isfinite(x)&np.isfinite(y)
    if control is not None:
        z=np.broadcast_to(np.asarray(control,float)[:,None],y.shape)
        valid &= np.isfinite(z)
    xd,tx=detrend(x,years,valid); yd,ty=detrend(y,years,valid)
    extra={}
    valid_control=np.ones(y.shape[1],bool)
    if control is not None:
        # Frisch-Waugh-Lovell: residualize against year, then detrended control.
        # This equals two regressions on [intercept, actual year, control].
        zd,_=detrend(z,years,valid)
        extra['r_uncontrolled_matched']=pearson(xd,yd,valid)[0]
        extra['r_predictor_control']=pearson(xd,zd,valid)[0]
        extra['r_field_control']=pearson(yd,zd,valid)[0]
        scale=np.maximum(1.,np.max(np.where(valid,np.abs(z),0),axis=0))
        span=np.max(np.where(valid,np.abs(zd),0),axis=0)
        valid_control=span>64*np.finfo(float).eps*scale
        z0=np.where(valid,zd,0)
        denominator=(z0*z0).sum(0)
        bx=np.divide((np.where(valid,xd,0)*z0).sum(0),denominator,
            out=np.zeros(y.shape[1]),where=(denominator>0)&valid_control)
        by=np.divide((np.where(valid,yd,0)*z0).sum(0),denominator,
            out=np.zeros(y.shape[1]),where=(denominator>0)&valid_control)
        xd=xd-bx*zd; yd=yd-by*zd
    r,n=pearson(xd,yd,valid)
    # Reject constant or exactly linear series whose residual is floating point roundoff.
    for original,residual in ((x,xd),(y,yd)):
        scale=np.maximum(1.,np.max(np.where(valid,np.abs(original),0),axis=0))
        span=np.max(np.where(valid,np.abs(residual),0),axis=0)
        r[span <= 64*np.finfo(float).eps*scale]=np.nan
    r[(n<min_seasons)|~valid_control]=np.nan
    fitted_dof=3 if control is None else 4
    nominal=pvalue(r,n.astype(float)-fitted_dof)
    inflation=np.ones(n.shape,float)
    available=np.isfinite(r)
    positions={int(year):i for i,year in enumerate(years)}
    for lag in range(1,max_lag+1):
        pairs=[(i,positions[int(year)+lag]) for i,year in enumerate(years) if int(year)+lag in positions]
        if len(pairs)<3:
            available[:]=False
            continue
        a,b=np.array(pairs).T
        mask=valid[a]&valid[b]
        rx,npair=pearson(xd[a],xd[b],mask)
        ry,_=pearson(yd[a],yd[b],mask)
        good=(npair>=3)&np.isfinite(rx)&np.isfinite(ry)
        available &= good
        weight=np.divide(npair,n,out=np.zeros(n.shape,float),where=n>0)
        inflation += 2*np.where(good,weight*rx*ry,0.)
    available &= np.isfinite(inflation)&(inflation>0)
    neff=np.divide(n,inflation,out=np.full(n.shape,np.nan),where=available)
    neff=np.minimum(n,neff)
    dof=neff-fitted_dof
    result=dict(r=r,p=pvalue(r,dof),p_nominal=nominal,n=n,n_eff=neff,dof=dof,
                variance_inflation=inflation,field_trend=ty,predictor_trend=tx,**extra)
    return {key:values.reshape(shape) for key,values in result.items()}


def _as_float(value):
    return np.ma.asarray(value, dtype=float).filled(np.nan)


def _design(index, control, years):
    """Return a scaled joint design and the unchanged predictor scale."""
    tc = years - years.mean()
    xc = index - index.mean()
    if control is None:
        scales = np.array([np.max(np.abs(tc)), np.max(np.abs(xc))])
        if not np.isfinite(scales).all() or np.any(scales == 0):
            return None, np.nan
        design = np.column_stack((np.ones(len(years)), tc / scales[0],
                                  xc / scales[1]))
    else:
        # Preserve the original four-column arithmetic and ordering exactly.
        zc = control - control.mean()
        scales = np.array([np.max(np.abs(tc)), np.max(np.abs(zc)),
                           np.max(np.abs(xc))])
        if not np.isfinite(scales).all() or np.any(scales == 0):
            return None, np.nan
        design = np.column_stack((np.ones(len(years)), tc / scales[0],
                                  zc / scales[1], xc / scales[2]))
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 64 * np.finfo(float).eps * singular[0] * max(design.shape):
        return None, np.nan
    return design, scales[-1]


def partial_regression(predictor, field, years, control=None,
                       min_years=10, max_lag=3):
    """Fit intercept + actual year [+ control] + beta * predictor per cell.

    Each cell uses its own paired finite field years. The predictor and the
    optional control must be finite over the configured year vector and
    independent after removal of the linear year trend. Their regression
    residuals are never standardized again; beta is per original predictor
    unit. A constant response has beta=0 even when correlation is undefined,
    and arbitrarily small coefficients are retained.

    Bartlett autocorrelations use actual-year lag pairs. Effective and nominal
    degrees of freedom are Neff-k and n-k, where k=3 without a control and
    k=4 with a control. No spatial significance selection is applied.
    """
    x = _as_float(predictor)
    year = _as_float(years)
    z = None if control is None else _as_float(control)
    yfield = _as_float(field)
    if (x.ndim != 1 or year.shape != x.shape
            or (z is not None and z.shape != x.shape)
            or yfield.ndim < 2 or yfield.shape[0] != len(x)):
        raise ValueError('Predictor, optional control and years must be matching '
                         '1-D arrays and match the field time axis')
    if (not np.isfinite(x).all() or not np.isfinite(year).all()
            or (z is not None and not np.isfinite(z).all())):
        raise ValueError('Predictor, optional control and years must be finite')
    if (len(np.unique(year)) != len(year)
            or not np.all(year == np.rint(year)) or np.any(np.diff(year) <= 0)):
        raise ValueError('Season years must be unique increasing integer model/calendar years')
    parameters = 3 if z is None else 4
    if (isinstance(min_years, (bool, np.bool_))
            or not isinstance(min_years, (int, np.integer))
            or min_years < parameters
            or isinstance(max_lag, (bool, np.bool_))
            or not isinstance(max_lag, (int, np.integer))
            or max_lag < 0 or len(year) < parameters):
        raise ValueError(f'Require >= {parameters} years, integer '
                         f'min_years >= {parameters}, and integer max_lag >= 0')
    design, _ = _design(x, z, year)
    if design is None:
        raise ValueError('Predictor and optional control must be nonconstant '
                         'and independent after detrending')
    result = correlate(x, yfield, year, min_seasons=min_years,
                       max_lag=max_lag, control=z)
    shape = yfield.shape[1:]
    y = yfield.reshape(len(year), -1)
    valid = np.isfinite(y)
    beta = np.full(y.shape[1], np.nan)
    complete = valid.all(axis=0)
    if len(year) >= min_years and np.any(complete):
        complete_y = y[:, complete]
        coefficients = np.linalg.lstsq(
            design, complete_y - complete_y.mean(axis=0), rcond=None)[0]
        _, xscale = _design(x, z, year)
        beta[complete] = coefficients[-1] / xscale
    incomplete_columns = np.flatnonzero(~complete)
    patterns, inverse = np.unique(valid[:, incomplete_columns].T,
                                  axis=0, return_inverse=True)
    for group, pattern in enumerate(patterns):
        if pattern.sum() < max(parameters, min_years):
            continue
        paired_control = None if z is None else z[pattern]
        paired_design, xscale = _design(x[pattern], paired_control, year[pattern])
        if paired_design is None:
            continue
        columns = incomplete_columns[inverse == group]
        paired_y = y[np.ix_(pattern, columns)]
        coefficients = np.linalg.lstsq(
            paired_design, paired_y - paired_y.mean(axis=0), rcond=None)[0]
        beta[columns] = coefficients[-1] / xscale
    result['beta'] = beta.reshape(shape)
    return result
