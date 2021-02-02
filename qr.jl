using LinearAlgebra

function Norm(v::Vector{T}) where T <: Number
  map(x -> x^2, v) |> sum |> sqrt
end

function SpecialSign(r::T) where T <: Number
  r == 0 ? 1 : sign(r)
end

function stdBasis(size::Int, n::Int)
  v = zeros(Int, size)
  v[n] = 1
  return v
end

function HouseholderReflect(A::Array{T}) where T <: Number
  a1 = A[:, 1]
  s = size(a1)[1]
  v1 = a1 + SpecialSign(a1[1]) * Norm(a1) * stdBasis(s, 1)
  H = I(s) - 2 * (v1 * transpose(v1)) / (transpose(v1) * v1)
  return H
end

function HouseholderQRAux(A::Array{Array})
  l = Array{Number}[]
  if min(size(A[1])...) == 0
    return A[2:size(A)[1]]
  else
    s = size(A[1])
    H = HouseholderReflect(A[1])
    newA = vcat(Array[(H * A[1])[2:s[1], 2:s[2]]], A[2:size(A)[1]], Array[H])
    HouseholderQRAux(newA)
  end
end

function FillHouseholderQR(A::Array{T}, s::Int) where T <: Number 
  Q = Matrix{T}(I(s))
  ind = s - size(A)[1] + 1
  Q[ind:s, ind:s] .= A
  return Q
end

function HouseholderReflectors(A::Array{T}, returnR::Bool=false) where T <: Number
  arr = HouseholderQRAux(Array[A])
  s = size(arr[1])[1]
  H = map(x -> FillHouseholderQR(x, s), arr)
  returnR ? reduce(*, reverse(H)) * A : H
end
