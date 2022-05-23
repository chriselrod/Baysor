using Random

function sample_position_params!(data::BmmData, shape_prior::ShapePrior{N})::MvNormalF{N} where N
    μ = sample_center!(data, Val{N}()) # TODO: should we store it as non-static arrays from the beginning?
    Σ = zero(MMatrix{N,N,Float64})
    d = shuffle(sample_var(shape_prior))
    for n = 1:N
      Σ[n,n] = d[n]
    end
    return MvNormalF(μ, Σ)
end

function sample_center!(data::BmmData, ::Val{N}; cache_size::Int=10000) where {N}
    if length(data.center_sample_cache) == 0 # otherwise, sampling takes too long
        data.center_sample_cache = sample(1:size(data.x, 1), Weights(confidence(data)), cache_size)
    end
    i = pop!(data.center_sample_cache)
    ret = MVector{N,Float64}(undef)
    pd = position_data(data)
    for n = 1:N
        ret[n] = pd[n,i]
    end
    return ret
end

function sample_composition_params(data::BmmData, N)
    gene_counts = shuffle(sample(@view(data.components[begin:N])).composition_params.counts)
    return CategoricalSmoothed(gene_counts, smooth=data.distribution_sampler.composition_params.smooth, sum_counts=sum(gene_counts));
end

function sample_distribution!(data::BmmData, N = length(data.components))
    data.max_component_guid += 1
    shape_prior = data.distribution_sampler.shape_prior
    position_params = sample_position_params!(data, shape_prior);
    composition_params = sample_composition_params(data, N);

    return Component(position_params, composition_params; shape_prior=deepcopy(shape_prior), guid=data.max_component_guid);
end
