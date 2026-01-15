using Plots, StatsPlots
using CSV
using DataFrames

# Load the data. Columns are operation, size, implementation, time
df1 = DataFrame(CSV.File("out/results_lazy.csv"))
# columns :info :median_time :augmented_size 

# Convert size from int to float in millions, 4 decimal places
#df1.augmented_size .= round.(df.augmented_size ./ 1e6; digits=5)

# Plot the size of the augmented graph against the time taken, annotated by implementation
@df df1 scatter(:median_time, :augmented_size, group = :info,
	title = "Augmented Graph size and time \n for different optimization strategies",
	xlabel = "Time (s)",
	ylabel = "Augmented Graph Size (millions of edges)",
	xlimit = (18, 40),
	ylimit = (1.5, 1.85),
	
	series_annotations = (text.(df1.info, :center, 8)),
	marker = (35, 0.2, :orange),
	size=(700,500),
	# no legend
	legend = false
	)
display(current())
savefig("out/strategies.png")


df2 = DataFrame(CSV.File("out/results_stops2.csv"))
df2.augmented_size .= round.(df2.augmented_size ./ 1e6; digits=5)

# Plot the size of the augmented graph and the time taken, on 2 separete y axes
# x axis is the number of stops

df2.nb_stops = df2.info
# Plot the time and recomputed time vs number of stops
@df df2 plot(:nb_stops, :median_time,
	title = "Augmented Graph size and time \n for different number of stops during CH construction",
	xlabel = "Number of complete re-computations during CH construction",
	ylabel = "Time (s)",
	size=(700,500),
	label = "Construction Time",
	marker = (:circle,6),
	)
# Add recompute time
plot!(df2.nb_stops, df2.median_time_recompute,
	label = "Recomputed CH Time",
	marker = (:star5,6),
	color=:green,
)

# Add a second y axis for Augmented Size 
plot!(twinx(),df2.nb_stops, df2.augmented_size,
	label = "Augmented Graph Size",
	ylabel = "Augmented Graph Size (millions of edges)",
	marker = (:diamond,6),
	color = :red,
	linestyle = :dash,
	ylimit = (1.0, 1.85),

)

display(current())
savefig("out/stops.png")

df3 = DataFrame(CSV.File("out/results_hops2.csv"))
# Plot the size of the augmented graph and the time taken, on 2 separete y axes
# x axis is the max hops for witness search
df3.max_hops = df3.info
@df df3 plot(:max_hops, :median_time,
	title = "Augmented Graph size and time \n for different max hops in witness search",
	xlabel = "Max Hops in Witness Search",
	ylabel = "Time (s)",
	size=(700,500),
	label = "Construction Time",
	marker = (:circle,6),
	)
	# Add recompute time
plot!(df3.max_hops, df3.median_time_recompute,
	label = "Recomputed CH Time",
	marker = (:star5,6),
	color=:green,
)
# Add augmented size
plot!(twinx(),df3.max_hops, df3.augmented_size,
	label = "Augmented Graph Size",
	ylabel = "Augmented Graph Size (millions of edges)",
	marker = (:diamond,6),
	color = :red,
	linestyle = :dash,
)
display(current())
savefig("out/hops.png")

# Plot the results for different strategies
df4 = DataFrame(CSV.File("out/results_strategies2.csv"))
df4.augmented_size .= round.(df4.augmented_size ./ 1e6; digits=5)
# Plot the size of the augmented graph and the time taken, on 2 separete y axes
# x axis is the max hops for witness search
x_axis = 1:nrow(df4)
@df df4 plot(x_axis, :median_time,
	title = "Augmented Graph size and time \n for different optimization strategies",
	xlabel = "Strategy Index",
	ylabel = "Time (s)",
	ylimit = (8, 30),
	size=(700,500),
	label = "Construction Time",
	marker = (:circle,6),
	xticks = (x_axis, df4.info),
	legend = :topleft,
	)
	# Add recompute time
plot!(x_axis, df4.median_time_recompute,
	label = "Recomputed CH Time",
	marker = (:star5,6),
	color=:green,
)
# Add augmented size
plot!(twinx(),x_axis, df4.augmented_size,
	label = "Augmented Graph Size",
	ylabel = "Augmented Graph Size (millions of edges)",
	ylimit = (1.52, 1.6),
	marker = (:diamond,6),
	color = :red,
	linestyle = :dash,
)
display(current())
savefig("out/strategies2.png")

######################################################
# PHAST results
######################################################


df5 = DataFrame(CSV.File("out/results_real_phast.csv"))
# Filter to only NY dataset
df6 = filter(row -> row.name == "NY", df5)
# Plot runtime vs nsources for real-world graph
# Columns : name, nsources, time_cpu, time_gpu, time_ref
@df df6 plot(:nsources, :time_cpu, 

	title = "PHAST Shortest Path Time vs Number of Sources \n for Real-world Graph (NY)",
	xlabel = "Number of Sources",
	ylabel = "PHAST Shortest Path Time (ms)",
	xscale = :log2,
	yscale = :log10,
	xlimit = [minimum(df6.nsources)*0.8, maximum(df6.nsources)*1.5],
	# Display xticks as integers
	xticks = (df6.nsources, string.(df6.nsources)),
	size=(700,500),
	label = "CPU PHAST",
	marker = (:circle,8),
	legend = :topleft,
	
	)
	# Add GPU and reference times
	@df df6 plot!(:nsources, :time_gpu, label = "GPU PHAST", marker = (:diamond,8))
	@df df6 plot!(:nsources, :time_ref, label = "Reference Dijkstra", marker = (:star5,8))

display(current())
savefig("out/real_phast_ny.png")

# Plot the speedup of CPU and GPU PHAST over reference Dijkstra
df6.speedup_cpu = df6.time_ref ./ df6.time_cpu
df6.speedup_gpu = df6.time_ref ./ df6.time_gpu

@df df6 plot(:nsources, :speedup_cpu,
	title = "Speedup of PHAST over Reference Dijkstra \n for Real-world Graph (NY)",
	xlabel = "Number of Sources",
	ylabel = "Speedup",
	ylimit = (0, maximum(vcat(df6.speedup_cpu, df6.speedup_gpu))*1.1),
	xscale = :log2,
	xticks = (df6.nsources, string.(df6.nsources)),
	size = (700,500),
	label = "CPU PHAST Speedup",
	marker = (:circle,8),
	legend = :topleft,
	)
	# Add GPU speedup
	@df df6 plot!(:nsources, :speedup_gpu, label = "GPU PHAST Speedup", marker = (:diamond,8))
display(current())
savefig("out/real_phast_speedup_ny.png")


# Do the same plots again with the added pre-processing time for CPU and GPU PHAST
preprocess = 10.230338 # seconds
# Drop the first 3 rows
df7 = df6[4:end, :]
@df df7 plot(:nsources, :time_cpu .+ preprocess * 1e3,
	title = "PHAST Shortest Path Time vs Number of Sources \n for Real-world Graph (NY) including Preprocessing Time",
	xlabel = "Number of Sources",
	ylabel = "PHAST Shortest Path Time (ms)",
	xscale = :log2,
	yscale = :log10,
	xlimit = [minimum(df7.nsources), maximum(df7.nsources)*2],
	# Display xticks as integers
	xticks = (df7.nsources, string.(df7.nsources)),
	size=(700,500),
	label = "CPU PHAST (with preprocessing)",
	marker = (:circle,8),
	legend = :bottomright,
	
	)
	# Add GPU and reference times
	@df df7 plot!(:nsources, :time_gpu .+ preprocess * 1e3, label = "GPU PHAST (with preprocessing)", marker = (:diamond,8))
	@df df7 plot!(:nsources, :time_ref, label = "Reference Dijkstra", marker = (:star5,8))

	
display(current())
savefig("out/real_phast_with_preprocessing_ny.png")

# Plot the speedup of CPU and GPU PHAST over reference Dijkstra including preprocessing time
df7.speedup_cpu = df7.time_ref ./ (df7.time_cpu .+ preprocess * 1e3)
df7.speedup_gpu = df7.time_ref ./ (df7.time_gpu .+ preprocess * 1e3)

@df df7 plot(:nsources, :speedup_cpu,
	title = "Speedup of PHAST over Reference Dijkstra \n for Real-world Graph (NY) including Preprocessing Time",
	xlabel = "Number of Sources",
	ylabel = "Speedup",
	xscale = :log2,
	xticks = (df7.nsources, string.(df7.nsources)),
	size = (700,500),
	label = "CPU PHAST Speedup (with preprocessing)",
	marker = (:circle,8),
	legend = :topright,
	)
	# Add GPU speedup
	@df df7 plot!(:nsources, :speedup_gpu, label = "GPU PHAST Speedup (with preprocessing)", marker = (:diamond,8))

	# Add the line at y=1
	hline!([1.0], linestyle = :dash, color = :black, label = "Break even")
display(current())
savefig("out/real_phast_speedup_with_preprocessing_ny.png")

# Repeat the same for the BAY dataset
df8 = filter(row -> row.name == "BAY", df5)
# Plot runtime vs nsources for real-world graph
# Plot runtime vs nsources for real-world graph
# Columns : name, nsources, time_cpu, time_gpu, time_ref
@df df8 plot(:nsources, :time_cpu, 

	title = "PHAST Shortest Path Time vs Number of Sources \n for Real-world Graph (BAY)",
	xlabel = "Number of Sources",
	ylabel = "PHAST Shortest Path Time (ms)",
	xscale = :log2,
	yscale = :log10,
	xlimit = [minimum(df8.nsources)*0.8, maximum(df8.nsources)*1.5],
	# Display xticks as integers
	xticks = (df8.nsources, string.(df8.nsources)),
	size=(700,500),
	label = "CPU PHAST",
	marker = (:circle,8),
	legend = :topleft,
	
	)
	# Add GPU and reference times
	@df df8 plot!(:nsources, :time_gpu, label = "GPU PHAST", marker = (:diamond,8))
	@df df8 plot!(:nsources, :time_ref, label = "Reference Dijkstra", marker = (:star5,8))

display(current())
savefig("out/real_phast_bay.png")

# Plot the speedup of CPU and GPU PHAST over reference Dijkstra
df8.speedup_cpu = df8.time_ref ./ df8.time_cpu
df8.speedup_gpu = df8.time_ref ./ df8.time_gpu
@df df8 plot(:nsources, :speedup_cpu,
	title = "Speedup of PHAST over Reference Dijkstra \n for Real-world Graph (BAY)",
	xlabel = "Number of Sources",
	ylabel = "Speedup",
	ylimit = (0, maximum(vcat(df8.speedup_cpu, df8.speedup_gpu))*1.1),
	xscale = :log2,
	xticks = (df8.nsources, string.(df8.nsources)),
	size = (700,500),
	label = "CPU PHAST Speedup",
	marker = (:circle,8),
	legend = :topleft,
	)
	# Add GPU speedup
	@df df8 plot!(:nsources, :speedup_gpu, label = "GPU PHAST Speedup", marker = (:diamond,8))
display(current())
savefig("out/real_phast_speedup_bay.png")


# Do the same plots again with the added pre-processing time for CPU and GPU PHAST
preprocess = 7.536501 # seconds
# Filter rows for nsources >= 8
df8 = df8[df8.nsources .>= 8, :]
@df df8 plot(:nsources, :time_cpu .+ preprocess * 1e3,
	title = "PHAST Shortest Path Time vs Number of Sources \n for Real-world Graph (BAY) including Preprocessing Time",
	xlabel = "Number of Sources",
	ylabel = "PHAST Shortest Path Time (ms)",
	xscale = :log2,
	yscale = :log10,
	xlimit = [minimum(df8.nsources), maximum(df8.nsources)*2],
	# Display xticks as integers
	xticks = (df8.nsources, string.(df8.nsources)),
	size=(700,500),
	label = "CPU PHAST (with preprocessing)",
	marker = (:circle,8),
	legend = :bottomright,
	
	)
	# Add GPU and reference times
	@df df8 plot!(:nsources, :time_gpu .+ preprocess * 1e3, label = "GPU PHAST (with preprocessing)", marker = (:diamond,8))
	@df df8 plot!(:nsources, :time_ref, label = "Reference Dijkstra", marker = (:star5,8))

	
display(current())
savefig("out/real_phast_with_preprocessing_bay.png")
# Plot the speedup of CPU and GPU PHAST over reference Dijkstra including preprocessing time
df8.speedup_cpu = df8.time_ref ./ (df8.time_cpu .+ preprocess * 1e3)
df8.speedup_gpu = df8.time_ref ./ (df8.time_gpu .+ preprocess * 1e3)

@df df8 plot(:nsources, :speedup_cpu,
	title = "Speedup of PHAST over Reference Dijkstra \n for Real-world Graph (BAY) including Preprocessing Time",
	xlabel = "Number of Sources",
	ylabel = "Speedup",
	xscale = :log2,
	xticks = (df8.nsources, string.(df8.nsources)),
	size = (700,500),
	label = "CPU PHAST Speedup (with preprocessing)",
	marker = (:circle,8),
	legend = :topright,
	)
	# Add GPU speedup
	@df df8 plot!(:nsources, :speedup_gpu, label = "GPU PHAST Speedup (with preprocessing)", marker = (:diamond,8))

	# Add the line at y=1
	hline!([1.0], linestyle = :dash, color = :black, label = "Break even")
display(current())
savefig("out/real_phast_speedup_with_preprocessing_bay.png")

println("Plots saved in out/ directory.")

