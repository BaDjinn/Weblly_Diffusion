module.exports = {
	async onPreBuild({ utils }) {
		await utils.cache.restore("./models/sd-turbo");
	},

	async onPostBuild({ utils }) {
		await utils.cache.save("./models/sd-turbo");
	},
};
